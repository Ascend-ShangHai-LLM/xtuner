# Copyright (c) OpenMMLab. All rights reserved.

from typing import Annotated, cast

import torch
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import get_logger, get_device

from ..linear import build_linear
from .attn_outputs import AttnOutputs
from .causal_conv1d import causal_conv1d_triton


# Temporary solution: use separate function objects for each call site, Dynamo will cache them separately
def _all_to_all_conv_pre_qk(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_v(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_gb(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_out(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    class FusedRMSNormGated(FLA_FusedRMSNormGated):
        def forward(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
        ) -> torch.Tensor:
            weight = self.weight
            if isinstance(weight, DTensor):
                weight = weight.to_local()

            return rms_norm_gated(
                x,
                g,
                weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

except ImportError:
    FusedRMSNormGated = None  # type: ignore
    DEVICE = get_device()
    if DEVICE == "npu":
        from .chunk_gated_delta_rule_npu.flash_gated_delta_rule import flash_gated_delta_rule as chunk_gated_delta_rule
    else:
        chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

logger = get_logger()

def replace_conv1d(weight, bias, origin_conv1d):
    original_conv = origin_conv1d
    kernel_size = original_conv.kernel_size[0]
    stride = original_conv.stride[0]
    padding = original_conv.padding[0]
    dilation = original_conv.dilation[0]

    # 创建新的卷积层
    new_conv = nn.Conv1d(
        in_channels=weight.shape[0],        # 对应 query_weight 的第一维
        out_channels=weight.shape[0],       # 输出通道数与输入相同（因为分组卷积 groups=in_channels）
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=weight.shape[0],            # 每个通道独立处理
        bias=(bias is not None)  # 根据是否有 bias 决定
    )

    # 载入权重
    new_conv.weight.data = weight  # query_weight 形状为 [1024, 1, 4]

    # 如果有 bias
    if bias is not None:
        new_conv.bias.data = bias
    return new_conv

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def varlen_to_nonvarlen(cu_seqlens, *vars):
    B = len(cu_seqlens) - 1
    max_len = max(cu_seqlens[i+1] - cu_seqlens[i] for i in range(B))
    nonvarlen_vars = [torch.zeros((B, max_len.item(), *var.shape[2:]), dtype=var.dtype, device=var.device) for var in vars]
    for i in range(B):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        seq_len = end - start
        if seq_len > 0:
            for j in range(len(nonvarlen_vars)):
                nonvarlen_vars[j][i, :seq_len] = vars[j][0, start:end]
    return nonvarlen_vars

def nonvarlen_to_varlen(cu_seqlens, nonvarlen_var):
    B = len(cu_seqlens) - 1
    total_len = cu_seqlens[-1]

    # 创建结果张量列表
    # varlen_vars = []

    feature_dims = nonvarlen_var.shape[2:]
    # 创建变长张量
    varlen_var = torch.zeros(
        (1, total_len.item(), *feature_dims),
        dtype=nonvarlen_var.dtype,
        device=nonvarlen_var.device
    )
    # varlen_vars.append(varlen_var)

    # 填充数据
    for i in range(B):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        seq_len = end - start
        if seq_len > 0:
            varlen_var[0, start:end] = nonvarlen_var[i, :seq_len]

    return varlen_var



def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        weight = self.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        input_dtype = hidden_states.dtype
        import torch_npu
        hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, self.variance_epsilon)[0]
        hidden_states = hidden_states * F.silu(gate)

        return hidden_states


class GatedDeltaNetConfig(BaseModel):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="forbid")
    num_value_heads: Annotated[int, Parameter(group="attention")]
    num_key_heads: Annotated[int, Parameter(group="attention")]
    key_head_dim: Annotated[int, Parameter(group="attention")]
    value_head_dim: Annotated[int, Parameter(group="attention")]
    conv_kernel_dim: Annotated[int, Parameter(group="attention")]
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    rms_norm_eps: Annotated[float, Parameter(group="attention")]

    def build(
        self,
        hidden_size: int,
        float8_cfg: Float8Config | None = None,
        **kwargs,
    ) -> "GatedDeltaNet":
        return GatedDeltaNet(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
        )


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str,
        rms_norm_eps: float,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        self.name = f"layers.{layer_idx}.gate_deltanet"
        self.float8_cfg = float8_cfg

        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = hidden_act
        self.rms_norm_eps = rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.causal_conv1d_fn = causal_conv1d_fn
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        if FusedRMSNormGated is None:
            self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.rms_norm_eps)
        else:
            self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.rms_norm_eps, activation=self.activation)

        self.out_proj = build_linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            float8_cfg=self.float8_cfg,
        )

        self.in_proj_qkv = build_linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_z = build_linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_b = build_linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = build_linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward_for_sp(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
    ) -> AttnOutputs:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        # TODO: If full_graph mode is supported in the future, it needs to be modified to custom_op
        if seq_ctx.seq_idx is None:
            seq_idx = torch.cat(
                [
                    torch.full((s,), i, dtype=torch.int32, device=mixed_qkv.device)
                    for i, s in enumerate(seq_ctx.seq_lens_q)
                ],
                dim=0,
            )[None]
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        query, key, value = torch.split(
            mixed_qkv,  # (1, L/sp_size, 8192)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        # (1, L, 8192/sp_size)
        query = query.transpose(1, 2)  # (1, dim, L/sp_size)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = _all_to_all_conv_pre_qk(
            query,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        key = _all_to_all_conv_pre_qk(
            key,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        value = _all_to_all_conv_pre_v(
            value,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )

        # query =  (1, dim/sp_size, L)
        query_weight, key_weight, value_weight = torch.split(
            weight,  # (8192, 4)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )

        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_rank = seq_ctx.sequence_parallel_mesh.get_local_rank()
        sp_size = seq_ctx.sequence_parallel_mesh.size()
        query_weight = query_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        key_weight = key_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        value_weight = value_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]

        query = query.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn
        key = key.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn
        value = value.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn

        if True:
            if seq_ctx.cu_seq_lens_q is not None and seq_ctx.cu_seq_lens_q.device != query.device:
                # origin_device = seq_ctx.cu_seq_lens_q.device
                seq_ctx.cu_seq_lens_q = seq_ctx.cu_seq_lens_q.to(query.device)
            query = query.transpose(1, 2)  # (1, dim, L/sp_size)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            query, _ = causal_conv1d_triton(
                x=query,
                weight=query_weight,
                H=int(self.num_k_heads/sp_size),
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )
            key, _ = causal_conv1d_triton(
                x=key,
                weight=key_weight,
                H=int(self.num_k_heads/sp_size),
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )
            value, _ = causal_conv1d_triton(
                x=value,
                weight=value_weight,
                H=int(self.num_v_heads/sp_size),
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        A_log = A_log.to(query.device)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        # (1,key_dim/sp_size, L)
        # query = query.transpose(1, 2).reshape(
        #     batch_size, seq_len * sp_size, -1, self.head_k_dim
        # )  # (1, L, num_k_heads/sp_size, head_k_dim)
        # key = key.transpose(1, 2).reshape(
        #     batch_size, seq_len * sp_size, -1, self.head_k_dim
        # )  # (1, L, num_k_heads/sp_size, head_k_dim)
        # value = value.transpose(1, 2).reshape(
        #     batch_size, seq_len * sp_size, -1, self.head_v_dim
        # )  # (1, L, num_v_heads/sp_size, head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)

            g = _all_to_all_gb(
                g,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            beta = _all_to_all_gb(
                beta,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)
        
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_list,
            chunk_indices=seq_ctx.chunk_indices,
            chunk_indices_list=seq_ctx.chunk_indices_list 
        )
       
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            core_attn_out = _all_to_all_out(
                core_attn_out,  # (1, L, num_v_head/sp_size, head_dim)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        attn_outputs: AttnOutputs = {
            "projected_output": output,
        }
        return attn_outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
    ) -> AttnOutputs:
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx, position_embeddings)

        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        # TODO: If full_graph mode is supported in the future, it needs to be modified to custom_op
        if seq_ctx.seq_idx is None:
            seq_idx = torch.cat(
                [
                    torch.full((s,), i, dtype=torch.int32, device=mixed_qkv.device)
                    for i, s in enumerate(seq_ctx.seq_lens_q)
                ],
                dim=0,
            )[None]
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        # TODO: due to the limitation of scatter_dim=1 in ulysses_all_to_all,
        # the implementation is very inelegant and inefficient, and needs to be refactored in the future.
        if self.causal_conv1d_fn is not None:
            mixed_qkv = mixed_qkv.transpose(1, 2)
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,  # need non contiguous
                weight=weight,
                bias=bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)
        else:    
            if seq_ctx.cu_seq_lens_q is not None and seq_ctx.cu_seq_lens_q.device != mixed_qkv.device:
                seq_ctx.cu_seq_lens_q = seq_ctx.cu_seq_lens_q.to(mixed_qkv.device)
            mixed_qkv, _ = causal_conv1d_triton(
                x=mixed_qkv,
                weight=weight,
                H=2*self.num_k_heads + self.num_v_heads,
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.num_k_heads,
                self.num_k_heads,
                self.num_v_heads,
            ],
            dim=1,
        )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_list,
            chunk_indices=seq_ctx.chunk_indices,
            chunk_indices_list=seq_ctx.chunk_indices_list 
        )
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        attn_outputs: AttnOutputs = {
            "projected_output": output,
        }
        return attn_outputs

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> AttnOutputs: ...

    __call__ = nn.Module.__call__
