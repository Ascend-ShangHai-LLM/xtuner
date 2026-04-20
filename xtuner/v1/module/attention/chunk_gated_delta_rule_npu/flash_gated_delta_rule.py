# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import warnings
from typing import Optional, Dict
import math

import torch
import torch_npu

# from mindspeed.lite.ops.triton.l2norm import l2norm_bwd, l2norm_fwd
# from mindspeed.lite.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
# from mindspeed.lite.ops.triton.wy_fast import recompute_w_u_fwd
# from mindspeed.lite.ops.triton.solve_tril import solve_tril
# from mindspeed.lite.ops.triton.cumsum import chunk_local_cumsum
# from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from .triton_core.l2norm import l2norm_bwd, l2norm_fwd
from .triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

from .triton_core.solve_tril import solve_tril
from .triton_core.cumsum import chunk_local_cumsum
from .triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

def cdiv_torch(a, b):
    return (a + b - 1) // b

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
    chunk_indices_list: Dict[str, Optional[list[int]]] = None,
    chunk_size: int = 64,
):

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, head_first=False)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices[str(chunk_size)],
        chunk_size=chunk_size,
        output_dtype=torch.float32
    )

    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=k.dtype
    )
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()
    w, u = torch_npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )

    h, v_new, final_state = torch_npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial_state,
        cu_seqlens,
        chunk_indices[str(chunk_size)],
        output_final_state,
        chunk_size
    )


    o = torch_npu.npu_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        g,
        cu_seqlens,
        chunk_indices[str(chunk_size)],
        scale,
        chunk_size
    )
    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
    chunk_indices_list: Dict[str, Optional[list[int]]] = None,
    chunk_size: int = 64,
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    w, u = torch_npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        chunk_size=chunk_size,
    )


    do = do.transpose(1, 2).contiguous()
    
    h, v_new, _ = torch_npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial_state,
        cu_seqlens,
        chunk_indices[str(chunk_size)],
        False,
        chunk_size
    )
    

    dv = torch_npu.npu_chunk_bwd_dv_local(
      q, 
      k, 
      do, 
      g, 
      g_gamma=None, 
      A=A,
      cu_seqlens=cu_seqlens_list, 
      chunk_indices=chunk_indices_list[str(chunk_size)], 
      scale=scale, 
      chunk_size=chunk_size
    )
    
    dh, dh0, dv = torch_npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list, 
        chunk_indices=chunk_indices_list[str(chunk_size)], 
        scale=scale,
        chunk_size=chunk_size
    )
    dh0 = None

    dq, dk, dw, dg = torch_npu.npu_chunk_bwd_dqkwg(
        q, 
        k, 
        v_new, 
        g, 
        h, 
        do, 
        dh, 
        dv, 
        cu_seqlens_list, 
        chunk_indices_list[str(chunk_size)],  
        scale, 
        chunk_size
    )

    dA = torch_npu.npu_prepare_wy_repr_bwd_da(
        k, 
        v, 
        beta.float(), 
        A, 
        dw, 
        dv, 
        g.float(), 
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)], 
        chunk_size=chunk_size
    )

    dk2, dv, db, dg2 = torch_npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list[str(chunk_size)], 
        chunk_size=chunk_size
    )
    
    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(
            f"dg current type is {dg.dtype} , should be float32"
        )
    
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices_out=chunk_indices, head_first=False)
        
    return dq, dk, dv, db, dg, dh0
class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
        chunk_indices_list: Dict[str, Optional[list[int]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        chunk_indices = ctx.chunk_indices
        cu_seqlens_list = ctx.cu_seqlens_list
        chunk_indices_list = ctx.chunk_indices_list
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None, None


@torch.compiler.disable
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
    chunk_indices_list: Dict[str, Optional[list[int]]] = None,
    chunk_size: int = 64,
    head_first: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    cu_seqlens = cu_seqlens.to(torch.int64)
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype} , k current type is {k.dtype} ,v current type is {v.dtype} , they should are equal"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state