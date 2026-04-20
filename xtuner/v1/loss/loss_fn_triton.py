"""
ChunkLoss Triton 修复版本
解决所有功能问题:
1. 标签范围错误
2. UB 溢出问题 (分块处理)
3. 缺少 mask 处理
4. 数值稳定性问题
"""

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
from torch.autograd import Function

import triton.runtime.driver as driver
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]


# ============================================================================
# 大词汇表分块版本 (生产环境)
# ============================================================================

@triton.jit
def fused_ce_loss_kernel_chunked(
    logits_ptr,
    weight_ptr,
    labels_ptr,
    out_ptr,
    N,
    C,
    VOCAB_CHUNK_SIZE: tl.constexpr,  # 每块处理的类别数
    ignore_index: tl.constexpr,
):
    """
    大词汇表分块版本 (支持 vocab_size > 8K)

    核心思路:
    1. 将 vocab_size 分块处理
    2. 每块大小不超过 UB 容量
    3. 逐块计算 softmax 的 max 和 sum
    4. 最后计算 loss

    UB 容量规划:
    - 每块大小: VOCAB_CHUNK_SIZE * 2 bytes (FP16)
    - 推荐: 8192 * 2 = 16 KB < 85 KB (安全)
    """
    i = tl.program_id(0)
    if i >= N:
        return

    # 加载标签和权重
    label = tl.load(labels_ptr + i)
    weight = tl.load(weight_ptr + i)

    # 忽略指定索引
    if label == ignore_index:
        tl.store(out_ptr + i, 0.0)
        return

    # 分块计算 softmax
    # 初始化全局 max 和 sum
    global_max = float('-inf')
    global_sum = 0.0

    # 块偏移量
    chunk_offsets = tl.arange(0, VOCAB_CHUNK_SIZE)

    # 逐块处理
    num_chunks = tl.cdiv(C, VOCAB_CHUNK_SIZE)
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * VOCAB_CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + VOCAB_CHUNK_SIZE, C)

        # 当前块的 mask
        chunk_mask = (chunk_start + chunk_offsets) < C

        # 加载当前块的 logits
        chunk_logits = tl.load(
            logits_ptr + i * C + chunk_start + chunk_offsets,
            mask=chunk_mask,
            other=float('-inf')
        )

        # 计算当前块的 max
        chunk_max = tl.max(chunk_logits, 0)

        # 更新全局 max
        new_global_max = tl.maximum(global_max, chunk_max)

        # 计算当前块的 exp sum (考虑 max 的变化)
        if global_max == float('-inf'):
            # 第一块
            chunk_sum = tl.sum(tl.exp(chunk_logits - new_global_max), 0)
        else:
            # 后续块,需要调整之前的 sum
            # sum_new = sum_old * exp(old_max - new_max) + sum_chunk
            adjust = tl.exp(global_max - new_global_max)
            chunk_sum = global_sum * adjust + tl.sum(tl.exp(chunk_logits - new_global_max), 0)

        global_max = new_global_max
        global_sum = chunk_sum

    # 计算 log_sum_exp
    log_sum_exp = tl.log(global_sum)

    # 计算目标 logit
    # 找到 label 所在的块
    label_chunk_id = label // VOCAB_CHUNK_SIZE
    label_chunk_start = label_chunk_id * VOCAB_CHUNK_SIZE
    label_offset = label - label_chunk_start

    # 加载 label 所在块的 logits
    label_chunk_offsets = tl.arange(0, VOCAB_CHUNK_SIZE)
    label_chunk_mask = (label_chunk_start + label_chunk_offsets) < C

    label_chunk_logits = tl.load(
        logits_ptr + i * C + label_chunk_start + label_chunk_offsets,
        mask=label_chunk_mask,
        other=float('0.0')
    )

    # 提取目标 logit
    target_logit = tl.sum(
        label_chunk_logits * tl.where(label_chunk_offsets == label_offset, 1.0, 0.0),
        0
    )

    # 计算 log_prob
    log_prob = (target_logit - global_max) - log_sum_exp

    # 计算损失
    loss = -log_prob * weight
    tl.store(out_ptr + i, loss)


# ============================================================================# PyTorch 包装接口 (支持自动微分)# ============================================================================
class FusedCrossEntropyLoss(Function):
    """
    融合交叉熵损失的自定义自动微分函数
    使用 Triton 内核实现前向和反向传播
    """
    
    @staticmethod
    def forward(ctx, logits, weight, labels, ignore_index=-100, vocab_chunk_size=4096):
        """
        前向传播
        
        Args:
            logits: [N, C] - logits 张量
            labels: [N] - 标签张量
            weight: [N] - 权重张量
            ignore_index: int - 忽略的标签索引
            vocab_chunk_size: int - 分块大小 (仅大词汇表使用)
            
        Returns:
            loss: scalar - 总损失
        """
        N, C = logits.shape
        assert labels.shape == (N,)
        assert weight.shape == (N,)
        
        # 保存需要反向传播的张量和参数
        ctx.save_for_backward(logits, weight, labels)
        ctx.ignore_index = ignore_index
        ctx.vocab_chunk_size = vocab_chunk_size
        
        # 分配输出张量
        out = torch.empty(N, dtype=torch.float32, device=logits.device)
        
        # 大词汇表: 使用分块版本
        # 确保 vocab_chunk_size 是 2 的幂
        VOCAB_CHUNK_SIZE = triton.next_power_of_2(vocab_chunk_size)
        grid = (N,)
        
        # 调用 Triton 内核
        fused_ce_loss_kernel_chunked[grid](
            logits,
            weight,
            labels,
            out,
            N=N,
            C=C,
            VOCAB_CHUNK_SIZE=VOCAB_CHUNK_SIZE,
            ignore_index=ignore_index,
        )
        
        # 返回总损失
        return out.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            grad_output: scalar - 上游梯度
            
        Returns:
            grad_logits: [N, C] - logits 的梯度
            grad_weight: [N] - weight 的梯度
            None: labels 没有梯度
            None: ignore_index 没有梯度
            None: vocab_chunk_size 没有梯度
        """
        # 取出保存的张量
        logits, weight, labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        
        # 初始化输出梯度
        grad_logits = torch.empty_like(logits, dtype=torch.bfloat16)
        grad_weight = torch.empty_like(weight, dtype=torch.bfloat16)
        
        # 形状
        N, V = logits.shape
        
        # 启动 Triton 内核
        BLOCK_V = min(4096, V)
        grid = (N,)
        
        chunk_loss_bw_kernel[grid](
            logits, labels, weight, grad_output,
            grad_logits, grad_weight,
            N, V, ignore_index,
            BLOCK_V=BLOCK_V
        )
        
        # 返回梯度
        return grad_logits, grad_weight, None, None, None


# 包装函数，方便使用
def fused_cross_entropy_loss(
    logits,
    weight,
    labels,
    ignore_index=-100,
    vocab_chunk_size=4096
):
    """
    融合交叉熵损失 (自动选择最优实现，支持自动微分)
    
    Args:
        logits: [N, C] - logits 张量
        labels: [N] - 标签张量
        weight: [N] - 权重张量
        ignore_index: int - 忽略的标签索引
        vocab_chunk_size: int - 分块大小 (仅大词汇表使用)
        
    Returns:
        loss: scalar - 总损失
    """
    return FusedCrossEntropyLoss.apply(logits, weight, labels, ignore_index, vocab_chunk_size)




@triton.jit
def chunk_loss_bw_kernel(
    logits, labels, loss_weight, grad_output,  # 输入
    grad_logits, grad_loss_weight,            # 输出
    N, V, ignore_index,                       # 形状/配置
    BLOCK_V: tl.constexpr                     # 分块大小
):
    # 每个线程处理 1 个样本 (2D logits 对应 N 个独立样本)
    pid = tl.program_id(0)
    if pid >= N:
        return

    # 读取当前样本的标签与权重
    label = tl.load(labels + pid)
    weight = tl.load(loss_weight + pid)
    go = tl.load(grad_output)  # 上游梯度标量

    # --------------------------
    # 忽略 ignore_index
    # --------------------------
    if label == ignore_index:
        # 梯度全部置 0
        for off in range(0, V, BLOCK_V):
            idx_v = off + tl.arange(0, BLOCK_V)
            mask = idx_v < V
            tl.store(grad_logits + pid * V + idx_v, 0.0, mask=mask)
        tl.store(grad_loss_weight + pid, 0.0)
        return

    # --------------------------
    # 1. 计算 softmax(logits)
    # --------------------------
    max_val = -float('inf')
    # 第一遍：找最大值（数值稳定）
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float('inf'))
        current_max = tl.max(l)  # 移除错误 mask 参数！
        max_val = tl.maximum(max_val, current_max)

    sum_exp = 0.0
    # 第二遍：计算 exp 和 sum_exp
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float('inf'))
        exp_l = tl.exp(l - max_val)
        sum_exp += tl.sum(exp_l)

    # --------------------------
    # 2. 计算 grad_logits = (softmax - one_hot) * weight * grad_output
    # --------------------------
    scale = weight * go
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float('inf'))
        prob = tl.exp(l - max_val) / sum_exp

        # scatter: label 位置减 1
        prob = tl.where(idx_v == label, prob - 1.0, prob)
        g = prob * scale
        tl.store(grad_logits + pid * V + idx_v, g, mask=mask)

    # --------------------------
    # 3. 计算 grad_loss_weight = ce_loss * grad_output
    # --------------------------
    ce_loss = -tl.log(tl.exp(tl.load(logits + pid * V + label) - max_val) / sum_exp)
    tl.store(grad_loss_weight + pid, ce_loss * go)


# --------------------------
# 主 ChunkLoss 类（前向 PyTorch / 反向 Triton）
# --------------------------
def fused_cross_entropy_loss_back(logits, loss_weight, labels, ignore_index, grad_output):
        # 初始化输出梯度

        grad_logits = torch.zeros_like(logits, dtype=torch.float32)
        grad_loss_weight = torch.zeros_like(loss_weight, dtype=torch.float32)

        # 形状
        N, V = logits.shape  # 2 维

        # 启动 Triton 内核
        BLOCK_V = min(1024, V)
        grid = (N,)

        chunk_loss_bw_kernel[grid](
            logits, labels, loss_weight, grad_output,
            grad_logits, grad_loss_weight,
            N, V, ignore_index,
            BLOCK_V=BLOCK_V
        )

        return grad_logits, grad_loss_weight

import torch
import torch.nn.functional as F
from torch.autograd import Function

class ChunkLoss(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, loss_weight: torch.Tensor, shifted_labels: torch.Tensor):
        # 保存所有需要反向传播的张量
        ctx.save_for_backward(logits, loss_weight, shifted_labels)
        ctx.ignore_index = -100

        # 计算损失
        logits = logits.float()
        ce_loss = F.cross_entropy(
            logits, shifted_labels,
            reduction="none",
            ignore_index=ctx.ignore_index
        )
        chunk_loss = (ce_loss * loss_weight).sum()

        return chunk_loss

    @staticmethod
    def backward(ctx, grad_output):
        # 取出正向保存的张量（顺序必须一致）

        print('grad_output', grad_output)
        logits, loss_weight, shifted_labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index

        # ====================== 1. 对 logits 的梯度 ======================
        probs = F.softmax(logits.float(), dim=-1)
        grad_logits = probs.clone()
        grad_logits.scatter_(dim=-1, index=shifted_labels.unsqueeze(-1), value=-1.0)

        mask = (shifted_labels != ignore_index).float()
        grad_logits *= loss_weight.unsqueeze(-1) * mask.unsqueeze(-1)
        grad_logits = grad_logits * grad_output

        # ====================== 2. 对 loss_weight 的梯度 ======================
        # 关键：loss_weight 的梯度 = 逐元素交叉熵 loss × mask × 上游梯度
        ce_loss = F.cross_entropy(
            logits, shifted_labels,
            reduction="none",
            ignore_index=ignore_index
        )
        grad_loss_weight = ce_loss * mask * grad_output

        # ====================== 返回顺序：logits, loss_weight, shifted_labels ======================
        return grad_logits, grad_loss_weight, None

