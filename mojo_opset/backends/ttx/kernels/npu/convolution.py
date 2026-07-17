# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao


from typing import Optional, Dict

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=['NUM_CHKS'])
def causal_conv1d_fwd_kernel_old(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0
        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

        b_y = tl.zeros((BT, BD), dtype=tl.float32)

        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (i_t * BT + i_w + tl.arange(0, BT)[:, None])

                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # We keep intra loop load because preloading will cause ub overflow under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])

                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (i_t * BT + i_w + tl.arange(0, BT)[:, None])
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = (i_t * BT + tl.arange(0, BT))
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w

                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d

                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                    tl.float32
                )

                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(
                residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
            )
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=['NUM_CHKS'])
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

        D_SUB: tl.constexpr = D // H
        HEADS_PER_BLOCK: tl.constexpr = BD // D_SUB
        b_y = tl.zeros((BT, BD), dtype=tl.float32)

        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (i_t * BT + i_w + tl.arange(0, BT)[:, None])

                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # We keep intra loop load because preloading will cause ub overflow under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])

                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (i_t * BT + i_w + tl.arange(0, BT)[:, None])
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = (i_t * BT + tl.arange(0, BT))
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w

                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d

                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                    tl.float32
                )

                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(
                residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
            )
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        head_start = (i_d * BD) // D_SUB
        batch_off = 0 if IS_VARLEN else i_b * T * D
        bos_off = bos if IS_VARLEN else 0
        p_y = tl.make_block_ptr(
            y + batch_off + head_start * D_SUB * T + bos_off * D_SUB,
            (T_len, HEADS_PER_BLOCK, D_SUB),
            (D_SUB, D_SUB * T, 1),
            (i_t * BT, 0, 0),
            (BT, HEADS_PER_BLOCK, D_SUB),
            (2, 1, 0),
        )
        b_y = tl.reshape(b_y, (BT, HEADS_PER_BLOCK, D_SUB))  # for better readability, no actual reshape
        tl.store(
            p_y,
            tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1, 2),
        )


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_state_fwd_boundary_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    initial_state,
    cu_seqlens,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUTPUT_HEAD_LAYOUT: tl.constexpr,
):
    """Overwrite only the sequence prefix whose convolution uses history."""
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_len = eos - bos
    else:
        bos = (i_n * T).to(tl.int64)
        T_len = T

    o_t = tl.arange(0, W)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    b_y = tl.zeros((W, BD), dtype=tl.float32)

    for i_w in tl.static_range(-W + 1, 1):
        source_t = o_t + i_w
        x_mask = (source_t >= 0)[:, None] & (source_t < T_len)[:, None] & m_d[None, :]
        state_slot = source_t + W
        state_mask = (source_t < 0)[:, None] & (state_slot >= 0)[:, None] & m_d[None, :]
        b_source = tl.load(
            x + (bos + source_t[:, None]) * D + o_d[None, :],
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)
        b_source += tl.load(
            initial_state + i_n * D * W + o_d[None, :] * W + state_slot[:, None],
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        b_weight = tl.load(
            weight + (i_w + W - 1) * D + o_d,
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        b_y += b_source * b_weight[None, :]

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)[None, :]
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * tl.sigmoid(b_y)
    if HAS_RESIDUAL:
        b_y += tl.load(
            residual + (bos + o_t[:, None]) * D + o_d[None, :],
            mask=(o_t < T_len)[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)

    output_mask = (o_t < T_len)[:, None] & m_d[None, :]
    if OUTPUT_HEAD_LAYOUT:
        K: tl.constexpr = D // H
        o_head = o_d // K
        o_k = o_d % K
        if IS_VARLEN:
            output_batch_off = 0
            output_time = bos + o_t[:, None]
        else:
            output_batch_off = i_n * H * T * K
            output_time = o_t[:, None]
        output_ptr = y + output_batch_off + o_head[None, :] * T * K + output_time * K + o_k[None, :]
    else:
        output_ptr = y + (bos + o_t[:, None]) * D + o_d[None, :]
    tl.store(
        output_ptr,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=output_mask,
    )


def _causal_conv1d_apply_initial_state(
    y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
    output_head_layout: bool,
) -> None:
    B, T, D = x.shape
    W = weight.shape[0]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BD = min(D, 128)
    causal_conv1d_state_fwd_boundary_kernel[(triton.cdiv(D, BD), N)](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        H=H,
        W=W,
        BD=BD,
        ACTIVATION=activation,
        OUTPUT_HEAD_LAYOUT=output_head_layout,
        multibuffer=False,
    )


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=['NUM_CHKS'])
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_Blk_D: tl.int32,
    NUM_CHKS: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_CHKS * NUM_Blk_D

    for task_id in range(pid, total_tasks, num_programs):
        i_d = task_id % NUM_Blk_D
        i_chk = task_id // NUM_Blk_D

        if IS_VARLEN:
            i_t = i_chk

            idx_chk = i_chk

            i_tg = idx_chk

            ptr = chunk_indices + idx_chk * 2
            i_n = tl.load(ptr).to(tl.int32)
            i_t_offset = tl.load(ptr + 1).to(tl.int32)

            i_t = i_t_offset

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)

            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_tg = i_chk

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        o_t = i_t * BT + tl.arange(0, BT)
        K: tl.constexpr = D // H
        HEADS_PER_BLOCK: tl.constexpr = BD // K
        head_start = (i_d * BD) // K
        o_head = o_d // K
        o_k = o_d % K
        if IS_VARLEN:
            dy_batch_off = 0
            dy_time_off = bos
        else:
            dy_batch_off = i_n * H * T * K
            dy_time_off = 0

        if HAS_WEIGHT:
            p_x = tl.make_block_ptr(x + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_x = tl.load(p_x, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")

        b_dx = tl.zeros((BT, BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)

        if not USE_FINAL_STATE:
            b_dw = tl.zeros((W, BD), dtype=tl.float32)

            # B T H K
            # B H T K
            p_dy = tl.make_block_ptr(
                dy + dy_batch_off + dy_time_off * K + head_start * T * K,
                (T_len, HEADS_PER_BLOCK, K),
                (K, T * K, 1),
                (i_t * BT, 0, 0),
                (BT * W, HEADS_PER_BLOCK, K),
                (2, 1, 0),
            )
            b_dy = tl.load(p_dy, boundary_check=(0, 1, 2)).to(tl.float32)
            b_dy = tl.reshape(b_dy, (BT * W, BD))

            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(
                    y + bos * D,
                    (T_len, D),
                    (D, 1),
                    (i_t * BT, i_d * BD),
                    (BT * W, BD),
                    (1, 0),
                )
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)

            for i_w in tl.static_range(0, W):
                b_dy_sub = tl.extract_slice(b_dy, [i_w, 0], [BT, BD], [1, 1])
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    b_y_sub = tl.extract_slice(b_y, [i_w, 0], [BT, BD], [1, 1])
                    b_ys = tl.sigmoid(b_y_sub)
                    b_dy_sub = b_dy_sub * b_ys * (1 + b_y_sub * (1 - b_ys))

                b_wdy = b_dy_sub
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw_sub = tl.sum(b_dy_sub * b_x, 0)
                    b_dw = tl.insert_slice(
                        b_dw,
                        b_dw_sub[None, :],
                        [W - i_w - 1, 0],
                        [1, BD],
                        [1, 1],
                    )

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_sub, 0)
                b_dx += b_wdy

            p_dw = tl.make_block_ptr(
                dw + i_tg * W * D,
                (W, D),
                (D, 1),
                (0, i_d * BD),
                (W, BD),
                (1, 0),
            )
            tl.store(p_dw, b_dw.to(dw.dtype.element_ty), boundary_check=(0, 1))

        elif i_t * BT >= W:
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(
                    dy + bos * K + head_start * T * K,
                    (T_len, HEADS_PER_BLOCK, K),
                    (K, T * K, 1),
                    (i_t * BT + i_w, 0, 0),
                    (BT, HEADS_PER_BLOCK, K),
                    (2, 1, 0),
                )
                b_dy = tl.load(p_dy, boundary_check=(0, 1, 2), padding_option="zero").to(tl.float32)
                b_dy = tl.reshape(b_dy, (BT, BD))
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(
                        dw + i_tg * W * D + (W - i_w - 1) * D + o_d,
                        b_dw.to(dw.dtype.element_ty),
                        mask=m_d,
                    )
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(
                    dy + bos * K + head_start * T * K,
                    (T_len, HEADS_PER_BLOCK, K),
                    (K, T * K, 1),
                    (i_t * BT + i_w, 0, 0),
                    (BT, HEADS_PER_BLOCK, K),
                    (2, 1, 0),
                )
                b_dy_shift = tl.load(p_dy, boundary_check=(0, 1, 2), padding_option="zero").to(tl.float32)
                b_dy_shift = tl.reshape(b_dy_shift, (BT, BD))
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                if HAS_WEIGHT:
                    b_dw = tl.sum(b_dy_shift * b_x, 0)

                    if USE_INITIAL_STATE:
                        mask_head_rows = o_t < i_w

                        p_dy_head = tl.make_block_ptr(
                            dy + bos * K + head_start * T * K,
                            (T_len, HEADS_PER_BLOCK, K),
                            (K, T * K, 1),
                            (i_t * BT, 0, 0),
                            (BT, HEADS_PER_BLOCK, K),
                            (2, 1, 0),
                        )
                        b_dy_head = tl.load(
                            p_dy_head, boundary_check=(0, 1, 2), padding_option="zero"
                        ).to(tl.float32)
                        b_dy_head = tl.reshape(b_dy_head, (BT, BD))
                        b_dy_head = tl.where(mask_head_rows[:, None], b_dy_head, 0.0)
                        if ACTIVATION == "swish" or ACTIVATION == "silu":
                            b_y_head = tl.load(
                                y + bos * D + o_t[:, None] * D + o_d,
                                mask=(mask_head_rows[:, None] & m_d[None, :]),
                                other=0.0,
                            ).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t

                        mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                        b_xc = tl.load(
                            initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                            mask=(mask_c[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)

                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(
                        dw + i_tg * W * D + (W - i_w - 1) * D + o_d,
                        b_dw.to(dw.dtype.element_ty),
                        mask=m_d,
                    )

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                b_wdy = (
                    b_dy_shift
                    if not HAS_WEIGHT
                    else (b_dy_shift * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1]))
                )
                b_dx += b_wdy

            if USE_INITIAL_STATE:
                p_dy0 = tl.make_block_ptr(
                    dy + bos * K + head_start * T * K,
                    (T_len, HEADS_PER_BLOCK, K),
                    (K, T * K, 1),
                    (i_t * BT, 0, 0),
                    (BT, HEADS_PER_BLOCK, K),
                    (2, 1, 0),
                )
                b_dy0 = tl.load(p_dy0, boundary_check=(0, 1, 2), padding_option="zero").to(tl.float32)
                b_dy0 = tl.reshape(b_dy0, (BT, BD))
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y0 = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                    b_y0 = tl.load(p_y0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
                    b_ys0 = tl.sigmoid(b_y0)
                    b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))

                for i_w in tl.static_range(1, W):
                    b_dh0_s = tl.zeros((BD,), dtype=tl.float32)
                    for i_k in tl.static_range(0, W):
                        m_rows = o_t == i_w - 1 - i_k
                        b_dy_sum = tl.sum(tl.where(m_rows[:, None], b_dy0, 0.0), 0)
                        if HAS_WEIGHT:
                            b_w_row = tl.sum(tl.extract_slice(b_w, [i_k, 0], [1, BD], [1, 1]), 0)
                            b_dh0_s += b_dy_sum * b_w_row
                        else:
                            b_dh0_s += b_dy_sum

                    tl.store(
                        dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                        b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
                        mask=m_d,
                    )

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if i_t * BT + BT >= T_len - W:
                start_tok = max(0, T_len - (W - 1))
                offset = i_t * BT + tl.arange(0, BT)
                tok_idx = offset - start_tok
                mask = (offset >= start_tok) & (offset < T_len)
                w_idx = 1 + tok_idx
                dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
                b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
                b_dx += b_dht

        p_dx = tl.make_block_ptr(dx + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.jit
def causal_conv1d_state_bwd_kernel(
    dy,
    y,
    weight,
    initial_state,
    dht,
    dh0,
    dw_state,
    cu_seqlens,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_len = eos - bos
        dy_batch_off = 0
        dy_time_off = bos
    else:
        bos = (i_n * T).to(tl.int64)
        T_len = T
        dy_batch_off = i_n * H * T * (D // H)
        dy_time_off = 0

    K: tl.constexpr = D // H
    o_t = tl.arange(0, W)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_head = o_d // K
    o_k = o_d % K
    m = (o_t[:, None] < T_len) & m_d[None, :]

    p_dy = (
        dy
        + dy_batch_off
        + o_head[None, :] * T * K
        + (dy_time_off + o_t[:, None]) * K
        + o_k[None, :]
    )
    b_dy = tl.load(p_dy, mask=m, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = tl.load(
            y + bos * D + o_t[:, None] * D + o_d[None, :], mask=m, other=0.0
        ).to(tl.float32)
        b_ys = tl.sigmoid(b_y)
        b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

    # State slot c contributes to output t with weight[c - 1 - t].
    tl.store(dh0 + i_n * D * W + o_d * W, 0.0, mask=m_d)
    for i_c in tl.static_range(1, W):
        b_dh = tl.zeros((BD,), dtype=tl.float32)
        for i_t in tl.static_range(0, W):
            if i_t < i_c:
                b_dy_t = tl.sum(tl.extract_slice(b_dy, [i_t, 0], [1, BD], [1, 1]), 0)
                b_w_t = tl.load(weight + (i_c - 1 - i_t) * D + o_d, mask=m_d, other=0.0).to(tl.float32)
                b_dh += b_dy_t * b_w_t
        if USE_FINAL_STATE:
            final_slot = i_c - T_len
            b_dh += tl.load(
                dht + i_n * D * W + o_d * W + final_slot,
                mask=m_d & (final_slot >= 0) & (final_slot < W),
                other=0.0,
            ).to(tl.float32)
        tl.store(dh0 + i_n * D * W + o_d * W + i_c, b_dh, mask=m_d)

    # Weight k receives state[c=k+1+t] at boundary output t.
    for i_k in tl.static_range(0, W):
        b_dw = tl.zeros((BD,), dtype=tl.float32)
        for i_t in tl.static_range(0, W):
            if i_t + i_k + 1 < W:
                b_dy_t = tl.sum(tl.extract_slice(b_dy, [i_t, 0], [1, BD], [1, 1]), 0)
                b_h = tl.load(
                    initial_state + i_n * D * W + o_d * W + i_t + i_k + 1,
                    mask=m_d,
                    other=0.0,
                ).to(tl.float32)
                b_dw += b_dy_t * b_h
        tl.store(dw_state + i_n * W * D + i_k * D + o_d, b_dw, mask=m_d)


@triton.jit
def causal_conv1d_final_state_bwd_kernel(
    dy,
    y,
    weight,
    dht,
    dx,
    cu_seqlens,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Recompute only the tail dx values that receive final-state gradients."""
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_len = eos - bos
        dy_batch_off = 0
        dy_time_off = bos
    else:
        bos = (i_n * T).to(tl.int64)
        T_len = T
        dy_batch_off = i_n * H * T * (D // H)
        dy_time_off = 0

    K: tl.constexpr = D // H
    o_r = tl.arange(0, W)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    num_tail_tokens = tl.minimum(T_len, W)
    token_start = tl.maximum(0, T_len - W)
    state_slot_start = tl.maximum(0, W - T_len)
    token_t = token_start + o_r
    token_mask = (o_r < num_tail_tokens)[:, None] & m_d[None, :]
    o_head = o_d // K
    o_k = o_d % K
    b_dx = tl.zeros((W, BD), dtype=tl.float32)

    for i_w in tl.static_range(0, W):
        output_t = token_t + i_w
        output_mask = token_mask & (output_t < T_len)[:, None]
        b_dy = tl.load(
            dy
            + dy_batch_off
            + o_head[None, :] * T * K
            + (dy_time_off + output_t[:, None]) * K
            + o_k[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = tl.load(
                y + (bos + output_t[:, None]) * D + o_d[None, :],
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            b_ys = tl.sigmoid(b_y)
            b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
        b_weight = tl.load(
            weight + (W - i_w - 1) * D + o_d,
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        b_dx += b_dy * b_weight[None, :]

    final_slot = state_slot_start + o_r
    b_dx += tl.load(
        dht + i_n * D * W + o_d[None, :] * W + final_slot[:, None],
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        dx + (bos + token_t[:, None]) * D + o_d[None, :],
        tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=token_mask,
    )


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_fwd_impl_old(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_origin: Dict[str, Optional[torch.LongTensor]] = None,
) -> torch.Tensor:
    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."
    B, T, D, W = *x.shape, weight.shape[0]

    NUM_CORES = get_num_cores()

    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    BD = 256
    if D < BD:
        BD = D
    assert D % BD == 0, "D must be divisible by BD."
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = chunk_indices_origin[str(BT)]
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None

        NUM_CHKS = triton.cdiv(T, BT) * B

    y = torch.empty_like(x)

    grid = (NUM_CORES,)

    causal_conv1d_fwd_kernel_old[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
        multibuffer=True,
    )

    if initial_state is not None:
        _causal_conv1d_apply_initial_state(
            y=y,
            x=x,
            weight=weight,
            H=1,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_head_layout=False,
        )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    return y.view(shape), final_state


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_origin: Dict[str, Optional[torch.LongTensor]] = None,
) -> torch.Tensor:
    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."
    B, T, D, W = *x.shape, weight.shape[0]

    NUM_CORES = get_num_cores()

    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    BD = 256
    if D < BD:
        BD = D
    assert D % BD == 0, "D must be divisible by BD."
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = chunk_indices_origin[str(BT)]
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None

        NUM_CHKS = triton.cdiv(T, BT) * B

    y = torch.empty_like(x)

    grid = (NUM_CORES,)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
        multibuffer=True,
    )

    if initial_state is not None:
        _causal_conv1d_apply_initial_state(
            y=y,
            x=x,
            weight=weight,
            H=H,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_head_layout=True,
        )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    return y.reshape(B,H,T,D//H), final_state


def causal_conv1d_bwd_impl(
    x: torch.Tensor,
    dy: torch.Tensor,
    H: int,
    dht: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_origin: Dict[str, Optional[torch.LongTensor]] = None,
):
    
    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."

    B, T, D = x.shape
    _,H,_,_ = dy.shape
    W = weight.shape[0] if weight is not None else None

    NUM_CORES = get_num_cores()
    # BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    BT = min(4, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    # Keep the full-sequence main kernel on the fast layout. State and final
    # state gradients are handled by small boundary kernels below.
    BD = 512
    if D < BD:
        BD = D
    assert D % BD == 0, "D must be divisible by BD."
    NUM_Blk_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = chunk_indices_origin[str(BT)]
        NUM_CHKS = len(chunk_indices)

        NT = len(chunk_indices)
    else:
        chunk_indices = None

        NT = triton.cdiv(T, BT)
        NUM_CHKS = NT * B

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd_impl_old(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
            chunk_indices_origin=chunk_indices_origin,
        )
    dx = torch.empty_like(x)
    # Some Ascend Triton vector stores may leave masked/aligned tail lanes
    # untouched.  Zero initialization makes the partial-gradient reduction
    # deterministic without changing the values written by the kernel.
    dw = weight.new_zeros(B * NT, W, D, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    if initial_state is not None:
        dh0 = initial_state.new_zeros(initial_state.shape)
        dw_state = weight.new_zeros(N, W, D, dtype=torch.float)
    else:
        dh0 = None
        dw_state = None

    grid = (NUM_CORES,)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state if initial_state is not None else x,
        dh0=dh0 if dh0 is not None else dx,
        dht=None,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        H=H,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        USE_INITIAL_STATE=False,
        NUM_Blk_D=NUM_Blk_D,
        NUM_CHKS=NUM_CHKS,
        multibuffer=False,
    )

    if initial_state is not None:
        state_BD = 128
        causal_conv1d_state_bwd_kernel[(triton.cdiv(D, state_BD), N)](
            dy=dy,
            y=y,
            weight=weight,
            initial_state=initial_state,
            dht=dht,
            dh0=dh0,
            dw_state=dw_state,
            cu_seqlens=cu_seqlens,
            T=T,
            D=D,
            H=H,
            W=W,
            BD=state_BD,
            ACTIVATION=activation,
            IS_VARLEN=cu_seqlens is not None,
            USE_FINAL_STATE=dht is not None,
            multibuffer=False,
        )

    if dht is not None:
        tail_BD = 128
        causal_conv1d_final_state_bwd_kernel[(triton.cdiv(D, tail_BD), N)](
            dy=dy,
            y=y if y is not None else x,
            weight=weight,
            dht=dht,
            dx=dx,
            cu_seqlens=cu_seqlens,
            T=T,
            D=D,
            H=H,
            W=W,
            BD=tail_BD,
            ACTIVATION=activation,
            IS_VARLEN=cu_seqlens is not None,
            multibuffer=False,
        )
    
    if weight is not None:
        dw = dw.sum(0).contiguous().to(weight)
        if dw_state is not None:
            dw = dw + dw_state.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    return dx.view(shape), dw, db, dr, dh0


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = eos - BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - BW + tl.arange(0, BW)
    m_t = o_t >= tl.maximum(bos, eos - W)
    m_d = o_d < D
    m_w = (o_w >= 0) & (o_w < W)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
    if USE_INITIAL_STATE:
        if T < BW:
            o_c = W - (BW - T) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(initial_state + i_n * D * W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
            b_x += b_cache

    tl.store(final_state + i_n * D * W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, D, W = *x.shape, state_len
    offsets = torch.arange(W, device=x.device, dtype=torch.long)

    if cu_seqlens is None:
        positions = T - W + offsets
        valid = positions >= 0
        safe_positions = positions.clamp(0, max(T - 1, 0))
        token_state = x.index_select(1, safe_positions).transpose(1, 2)
        valid = valid.view(1, 1, W).expand(B, 1, W)
        lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        N = B
    else:
        assert B == 1, "packed causal-conv state update expects batch size 1"
        boundaries = cu_seqlens.to(device=x.device, dtype=torch.long)
        starts, ends = boundaries[:-1], boundaries[1:]
        positions = ends[:, None] - W + offsets[None, :]
        valid = positions >= starts[:, None]
        safe_positions = positions.clamp(0, max(T - 1, 0)).reshape(-1)
        N = starts.numel()
        token_state = (
            x[0]
            .index_select(0, safe_positions)
            .reshape(N, W, D)
            .transpose(1, 2)
        )
        valid = valid[:, None, :]
        lengths = ends - starts

    if initial_state is None:
        return token_state.masked_fill(~valid, 0)

    assert initial_state.shape == (N, D, W), (
        f"initial_state must have shape {(N, D, W)}, got {tuple(initial_state.shape)}"
    )
    # For a segment shorter than W, prepend the tail of its incoming state.
    # Example: W=4 and length=1 produces [h1, h2, h3, x0].
    state_positions = lengths[:, None] + offsets[None, :]
    state_positions = state_positions.clamp(0, W - 1)
    history_state = torch.gather(
        initial_state,
        2,
        state_positions[:, None, :].expand(N, D, W),
    )
    return torch.where(valid, token_state, history_state)


@triton.jit()
def causal_conv1d_update_kernel_bdt_fwd(
    x_ptr,  # [B, D, T]
    conv_state_ptr,  # [B, D, ST]
    conv_state_update_ptr,
    weight_ptr,  # [D, W]
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,  # [B, D, OT]
    batch: tl.constexpr,
    dim: tl.constexpr,
    state_len: tl.constexpr,  # ST
    seq_len: tl.constexpr,  # T
    width: tl.constexpr,  # W
    out_len: tl.constexpr,  # OT
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    HAS_BIAS: tl.constexpr,  # TODO
    SILU_ACTIVATION: tl.constexpr,
    T_CHK_SIZE: tl.constexpr,
    D_CHK_SIZE: tl.constexpr,
    NUM_T_CHK: tl.constexpr,
    NUM_D_CHK: tl.constexpr,
    ST_STORE_HEAD_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    total_task = batch * NUM_D_CHK * NUM_T_CHK

    for task_id in tl.range(pid, total_task, pnum):
        di = task_id % NUM_D_CHK
        bti = task_id // NUM_D_CHK
        bi = bti // NUM_T_CHK
        ti = bti % NUM_T_CHK

        w = tl.load(
            tl.make_block_ptr(
                weight_ptr,
                shape=(dim, width),
                strides=(width, 1),
                offsets=(di * D_CHK_SIZE, 0),
                block_shape=(D_CHK_SIZE, width),
                order=(1, 0),
            ),
            boundary_check=(0, 1),
            padding_option="zero",
        )

        if ti == 0:
            st_b = tl.load(
                tl.make_block_ptr(
                    conv_state_ptr + bi * state_len * dim,
                    shape=(dim, state_len),
                    strides=(state_len, 1),
                    offsets=(di * D_CHK_SIZE, state_len - (width - 1)),
                    block_shape=(D_CHK_SIZE, (width - 1) + T_CHK_SIZE),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )
            offset0_x = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1_x = ti * T_CHK_SIZE + tl.arange(0, T_CHK_SIZE)
            mask_x = (offset0_x < dim)[:, None] & ((offset1_x >= 0) & (offset1_x < seq_len))[None, :]
            block_off_x = bi * dim * seq_len + offset0_x[:, None] * seq_len + offset1_x[None, :]
            x_b_tmp = tl.load(x_ptr + block_off_x, mask=mask_x, other=0)
            x_b = tl.insert_slice(st_b, x_b_tmp, (0, width - 1), (D_CHK_SIZE, T_CHK_SIZE), (1, 1))
        else:
            offset0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1 = ti * T_CHK_SIZE - (width - 1) + tl.arange(0, T_CHK_SIZE + width - 1)
            mask = (offset0 < dim)[:, None] & ((offset1 >= 0) & (offset1 < seq_len))[None, :]
            block_off = bi * dim * seq_len + offset0[:, None] * seq_len + offset1[None, :]
            x_b = tl.load(x_ptr + block_off, mask=mask, other=0)

        out_block = tl.zeros((T_CHK_SIZE, D_CHK_SIZE), dtype=x_ptr.dtype.element_ty)
        x_b = tl.trans(x_b, (1, 0))
        w = tl.trans(w, (1, 0))

        new_state_start_off = seq_len - state_len
        t_start_off = ti * T_CHK_SIZE - (width - 1)
        t_end_off = (ti + 1) * T_CHK_SIZE
        if t_end_off >= new_state_start_off:
            t_off = t_start_off - new_state_start_off
            if t_off < -(width - 1):
                # NOTE: In order to avoid use tl.maximum for negative offset,
                #       we pre-compute a fix head tile size (ST_STORE_HEAD_TILE_SIZE)
                #       to store the scene of negative address
                x_new_h = tl.extract_slice(x_b, (-t_off, 0), (ST_STORE_HEAD_TILE_SIZE, D_CHK_SIZE), (1, 1))
                x_new_h = tl.trans(x_new_h, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1_h = tl.arange(0, ST_STORE_HEAD_TILE_SIZE)[None, :]
                nst_mask_h = (nst_off_y0 < dim) & (nst_off_y1_h >= 0) & (nst_off_y1_h < state_len)
                block_ptr_h = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1_h
                tl.store(conv_state_update_ptr + block_ptr_h, x_new_h, mask=nst_mask_h)
            else:
                x_new_s = tl.extract_slice(x_b, (width - 1, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
                x_new_s = tl.trans(x_new_s, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1 = width - 1 + t_off + tl.arange(0, T_CHK_SIZE)[None, :]
                nst_mask = (nst_off_y0 < dim) & (nst_off_y1 >= 0) & (nst_off_y1 < state_len)
                block_ptr = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1
                tl.store(conv_state_update_ptr + block_ptr, x_new_s, mask=nst_mask)

        for owi in tl.range(0, width):
            new_x = tl.extract_slice(x_b, (owi, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
            w_chl_wi = tl.extract_slice(w, (owi, 0), (1, D_CHK_SIZE), (1, 1))
            x_mul_chl_wi = new_x * w_chl_wi
            out_block += x_mul_chl_wi
        out_block = tl.trans(out_block, (1, 0))

        if SILU_ACTIVATION:
            out_block = out_block * tl.sigmoid(out_block)
        tl.store(
            tl.make_block_ptr(
                out_ptr,
                shape=(batch, dim, out_len),
                strides=(dim * out_len, out_len, 1),
                offsets=(bi, di * D_CHK_SIZE, ti * T_CHK_SIZE),
                block_shape=(1, D_CHK_SIZE, T_CHK_SIZE),
                order=(2, 1, 0),
            ),
            out_block[None, :, :],
            boundary_check=(0, 1, 2),
        )


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_bdt_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[str] = None,
):
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    out = torch.empty_like(x)
    NUM_CORES = get_num_cores()

    T_CHK_SIZE = 256
    D_CHK_SIZE = 16

    assert T_CHK_SIZE >= width

    NUM_T_CHK = triton.cdiv(out.shape[-1], T_CHK_SIZE)
    NUM_D_CHK = triton.cdiv(dim, D_CHK_SIZE)
    conv_state_update = torch.empty_like(conv_state)

    # A const tile size variable to update negative address of conv state
    ST_STORE_HEAD_TILE_SIZE = width if (seqlen % T_CHK_SIZE) > width else (width - seqlen % T_CHK_SIZE) % T_CHK_SIZE
    causal_conv1d_update_kernel_bdt_fwd[(NUM_CORES, 1)](
        x,
        conv_state,
        conv_state_update,
        weight,
        bias,
        conv_state_indices,
        out,
        batch=int(batch),
        dim=int(dim),
        state_len=int(conv_state.shape[-1]),
        seq_len=int(x.shape[-1]),
        width=int(width),
        out_len=int(out.shape[-1]),
        x_batch_stride=x.stride()[0],
        conv_batch_stride=conv_state.stride()[0],
        out_batch_stride=out.stride()[0],
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        T_CHK_SIZE=T_CHK_SIZE,
        D_CHK_SIZE=D_CHK_SIZE,
        NUM_T_CHK=NUM_T_CHK,
        NUM_D_CHK=NUM_D_CHK,
        ST_STORE_HEAD_TILE_SIZE=int(ST_STORE_HEAD_TILE_SIZE),
    )
    conv_state.copy_(conv_state_update)
    if unsqueeze:
        out = out.squeeze(-1)
    return out
