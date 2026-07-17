# Copyright (c) OpenMMLab. All rights reserved.

from typing import Annotated, Callable, Literal, cast

import torch
from cyclopts import Parameter
from mmengine import is_installed
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from transformers.models.llama.modeling_llama import repeat_kv
from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops import AttnOpOutputs, attn_impl_mapping, flash_attn_varlen_func, get_apply_rotary_emb
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_logger

from ..linear import build_linear
from ..rms_norm import RMSNorm
from .attn_outputs import AttnOutputs
from .kv_cache import fill_paged_kv_cache
from xtuner.v1.utils.event_record import event_timer


logger = get_logger()


class MHAConfig(BaseModel):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="forbid")
    num_attention_heads: Annotated[int, Parameter(group="attention")]
    num_key_value_heads: int
    head_dim: Annotated[int, Parameter(group="attention")]
    dropout: Annotated[float, Parameter(group="attention")] = 0.0
    # casual: bool = True
    qkv_bias: Annotated[bool, Parameter(group="attention")] = False
    qk_norm: bool = False
    rms_norm_eps: float = 1e-06
    rms_norm_type: Literal["default", "zero_centered"] = "default"
    o_bias: Annotated[bool, Parameter(group="attention")] = False
    sliding_window: Annotated[int | None, Parameter(group="attention")] = -1
    with_sink: Annotated[bool, Parameter(group="attention")] = False
    with_gate: Annotated[bool, Parameter(group="attention")] = False
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if self.attn_impl == "flash_attention" and get_device() == "cuda":
            if not (is_installed("flash-attn") or is_installed("flash-attn-3")):
                logger.warning("flash-attn is not installed, using `flex_attention` instead.")
                self.attn_impl = "flex_attention"
        return self

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
    ) -> "MultiHeadAttention":
        return MultiHeadAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=rope_scaling_cfg,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )


@torch.library.custom_op("xtuner::paged_attention_decoding", mutates_args=())
def paged_attention_decoding(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

    bs = block_table.size(0)
    attn_outputs = cast(
        torch.Tensor,
        flash_attn_with_kvcache(
            query_states.transpose(1, 2).transpose(0, 1)[:bs],
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        ),
    )
    return attn_outputs


@paged_attention_decoding.register_fake
def paged_attention_decoding_fake(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
):
    bs = block_table.size(0)
    return torch.empty_like(query_states.transpose(1, 2).transpose(0, 1)[:bs])


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        *,
        head_dim: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.0,
        # casual: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        o_bias: bool = False,
        with_sink: bool = False,
        with_gate: bool = False,
        attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention",
        rope_scaling_cfg: RopeScalingConfig | None = None,
        float8_cfg: Float8Config | None = None,
        generate_config: GenerateConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        sliding_window: int = -1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.name = f"layers.{layer_idx}.self_attn"
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout
        # self.is_causal = casual
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_type = rms_norm_type
        self.o_bias = o_bias
        self.generate_config = generate_config
        self.float8_cfg = float8_cfg
        self.layer_idx = layer_idx
        self.with_gate = with_gate

        self.q_proj = build_linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim
            if not with_gate
            else self.num_attention_heads * self.head_dim * 2,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.k_proj = build_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.v_proj = build_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            float8_cfg=self.float8_cfg,
        )
        self.o_proj = build_linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.o_bias,
            float8_cfg=self.float8_cfg,
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps, type=self.rms_norm_type)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps, type=self.rms_norm_type)

        self.with_sink = with_sink
        if self.with_sink:
            self.sinks = nn.Parameter(torch.empty(self.num_attention_heads))

        self.window_size = (-1, -1)
        if layer_type == "sliding_attention":
            self.window_size = (sliding_window, sliding_window)

        fope_sep_head = rope_scaling_cfg.fope_sep_head if rope_scaling_cfg is not None else None
        enable_partial_rotary = (
            rope_scaling_cfg.partial_rotary_factor != 1.0 if rope_scaling_cfg is not None else False
        )
        self.apply_rotary_emb = get_apply_rotary_emb(fope_sep_head, enable_partial_rotary=enable_partial_rotary)  # type: ignore

        self.attn_impl_func: Callable[..., AttnOpOutputs] = attn_impl_mapping[attn_impl]  # type: ignore[assignment]

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_emb(query_states, key_states, cos, sin)

        # TODO: support sliding attention in prefilling
        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."
        fill_paged_kv_cache(
            key_states,
            value_states,
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            seq_ctx.cu_seq_lens_q,
            seq_ctx.cu_seq_lens_k,
            seq_ctx.max_length_q,
            seq_ctx.max_length_k,
            seq_ctx.block_table,
        )

        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1

        attn_output = cast(
            torch.Tensor,
            flash_attn_varlen_func(
                query_states.transpose(1, 2).squeeze(0),
                key_states.transpose(1, 2).squeeze(0),
                value_states.transpose(1, 2).squeeze(0),
                cu_seqlens_q=seq_ctx.cu_seq_lens_q,
                cu_seqlens_k=seq_ctx.cu_seq_lens_k,
                max_seqlen_q=seq_ctx.max_length_q,
                max_seqlen_k=seq_ctx.max_length_k,
                dropout_p=self.dropout,
                causal=True,
            ),
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        assert seq_ctx.block_table is not None
        assert self.layer_idx is not None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_emb(query_states, key_states, cos, sin)

        seq_lens_k = seq_ctx.seq_lens_k
        block_table = seq_ctx.block_table
        block_size = past_key_values[self.layer_idx][0].size(1)
        bs = block_table.size(0)

        assert seq_ctx.cu_seq_lens_k.numel() - 1 == bs, f"{seq_ctx.cu_seq_lens_k.numel()}, {bs}"

        _key_states = key_states.transpose(1, 2).squeeze(0)
        _value_states = value_states.transpose(1, 2).squeeze(0)

        block_index = block_table[:, 0] + (seq_lens_k[:bs] - 1) // block_size
        past_key_values[self.layer_idx][0][block_index, (seq_lens_k[:bs] - 1) % block_size] = _key_states
        past_key_values[self.layer_idx][1][block_index, (seq_lens_k[:bs] - 1) % block_size] = _value_states

        assert self.window_size == (-1, -1), "Sliding attention in prefilling is not supported yet."
        attn_output = paged_attention_decoding(
            query_states,
            past_key_values[self.layer_idx][0],
            past_key_values[self.layer_idx][1],
            seq_lens_k,
            block_table,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _prepare_serial_sp_kv_for_fa(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_ctx: SequenceContext,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """为当前 chunk 构造 FlashAttention 所需的带历史前缀 KV。

        返回：
            key_states: 按当前 Q 中 doc 顺序拼接的完整 KV
            value_states: 按当前 Q 中 doc 顺序拼接的完整 KV
            cu_seqlens_k: 与拼接后的 KV 对应的边界
            max_seqlen_k: 当前 batch 内最大的 KV 长度

        同时：
            若当前 chunk 最后一个 doc 跨越右边界，将该 doc 的当前 KV
            片段保存到 kvcache，供下一个 chunk 使用。
        """
        cache = seq_ctx.kvcache
        current_chunk_idx = cache.chunk_idx
        doc_ids = cache.chunk_doc_ids[current_chunk_idx]

        assert doc_ids, f"chunk {current_chunk_idx} has no document"
        assert len(doc_ids) + 1 == seq_ctx.cu_seq_lens_q.numel(), (
            f"doc count mismatch: {len(doc_ids)=}, "
            f"{seq_ctx.cu_seq_lens_q.numel()=}"
        )

        # 不能覆盖原始当前 KV；保存到 cache 的必须仅是当前 chunk 的 KV。
        current_key_states = key_states
        current_value_states = value_states
        q_cu = seq_ctx.cu_seq_lens_q

        full_k_parts: list[torch.Tensor] = []
        full_v_parts: list[torch.Tensor] = []
        k_lens: list[int] = []

        for local_doc_idx, doc_id in enumerate(doc_ids):
            local_start = int(q_cu[local_doc_idx])
            local_end = int(q_cu[local_doc_idx + 1])

            # 当前 chunk 中该 doc 对应的 KV。
            current_k = current_key_states[:, :, local_start:local_end, :]
            current_v = current_value_states[:, :, local_start:local_end, :]

            # 当前 chunk 的第一个 doc，才可能拥有来自前面 chunk 的 KV 前缀。
            if local_doc_idx == 0 and cache.history_k is not None:
                full_k = torch.cat([cache.history_k, current_k], dim=2)
                full_v = torch.cat([cache.history_v, current_v], dim=2)
            else:
                full_k = current_k
                full_v = current_v

            full_k_parts.append(full_k)
            full_v_parts.append(full_v)
            k_lens.append(full_k.size(2))

        # Q 的 doc 顺序未变，因此 KV 必须以相同 doc 顺序拼接。
        key_states = torch.cat(full_k_parts, dim=2)
        value_states = torch.cat(full_v_parts, dim=2)

        cu_seqlens_k = torch.tensor(
            [0, *k_lens],
            dtype=torch.int32,
            device=key_states.device,
        ).cumsum(dim=0).to(torch.int32)

        # 当前 chunk 最后一个 doc 是唯一可能跨越右边界、需要保存 cache 的 doc。
        last_local_doc_idx = len(doc_ids) - 1
        last_doc_id = doc_ids[-1]
        _, last_doc_end = cache.doc_ranges[last_doc_id]

        chunk_end = (current_chunk_idx + 1) * cache.chunk_size
        local_start = int(q_cu[last_local_doc_idx])
        local_end = int(q_cu[last_local_doc_idx + 1])
        current_last_k = current_key_states[:, :, local_start:local_end, :].clone()
        current_last_v = current_value_states[:, :, local_start:local_end, :].clone()
        if last_doc_end > chunk_end:
            if last_local_doc_idx == 0 and cache.history_k is not None:
                cache.output_k = torch.cat([cache.history_k, current_last_k], dim=2)
                cache.output_v = torch.cat([cache.history_v, current_last_v], dim=2)
            else:
                cache.output_k = current_last_k
                cache.output_v = current_last_v
        else:
            cache.output_k = current_last_k[:, :, :0, :]
            cache.output_v = current_last_v[:, :, :0, :]

        return key_states, value_states, cu_seqlens_k, max(k_lens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        """Forward pass for the Multi-Head Attention module.

        This method dispatches to specific forward implementations based on the
        attention context (training, prefilling, or decoding).

        Args:
            hidden_states (torch.Tensor): The input hidden states, typically of shape
                (batch_size, seq_len, hidden_size).
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Tuple containing
                positional embedding tensors for rotary position embeddings (cos, sin).
            seq_ctx (SequenceContext): Context information about the sequences being processed,
                containing metadata like sequence lengths and attention masks.
            past_key_values (list[list[torch.Tensor]] | None, optional): Cached key and value
                states from previous forward passes. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after attention computation and projection.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.with_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
        else:
            gate = None
            query_states = self.q_proj(hidden_states).view(hidden_shape)  # [b, seq,  n_head, head_dim]
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)  # [b, n_head, seq , head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = self.apply_rotary_emb(query_states, key_states, cos, sin)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            sp_size = seq_ctx.sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)
            query_states = ulysses_all_to_all(
                query_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            key_states = ulysses_all_to_all(
                key_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            value_states = ulysses_all_to_all(
                value_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )

        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1

        kwargs = {}
        if self.with_sink:
            if isinstance(self.sinks, DTensor):
                sinks = self.sinks.to_local()
            else:
                sinks = self.sinks
            kwargs["s_aux"] = sinks
        # [b, n_head, seq, head_dim]
        
        # event_timer.add_tensor_shape("mha_query", query_states)
        fa_lens = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
        if hasattr(seq_ctx, "kvcache") and seq_ctx.kvcache is not None:
            key_states, value_states, cu_seqlens_k, max_seqlen_k = self._prepare_serial_sp_kv_for_fa(
                key_states,
                value_states,
                seq_ctx,
            )
        else:
            cu_seqlens_k = seq_ctx.cu_seq_lens_k
            max_seqlen_k = seq_ctx.max_length_k
        # event_timer.add_tensor("cu_seq_lens", fa_lens)
        attn_op_outputs = self.attn_impl_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=seq_ctx.cu_seq_lens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_ctx.max_length_q,
            max_seqlen_k=max_seqlen_k,
            window_size=self.window_size,
            dropout_p=self.dropout,
            softmax_scale=self.scaling,
            causal=True,
            deterministic=XTUNER_DETERMINISTIC,
            **kwargs,
        )
        raw_output = attn_op_outputs["raw_output"]
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            raw_output = ulysses_all_to_all(
                raw_output,
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
        raw_output = raw_output.reshape(*input_shape, -1).contiguous()
        if self.with_gate:
            assert gate is not None
            raw_output = raw_output * torch.sigmoid(gate)

        projected_output = self.o_proj(raw_output)
        attn_outputs: AttnOutputs = {
            "projected_output": projected_output,
            **attn_op_outputs,
        }
        return attn_outputs

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.head_dim
        num_heads = self.num_key_value_heads

        generate_config = self.generate_config
        assert generate_config is not None, "Model configuration for generation is not set."

        max_length = max_length or generate_config.max_length
        block_size = block_size or generate_config.block_size
        max_batch_size = max_batch_size or generate_config.max_batch_size

        num_blocks = min(max_batch_size, max_length // block_size * max_batch_size)
        block_size = block_size or generate_config.block_size

        if generate_config.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {generate_config.dtype}")

        cache_k = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")
        cache_v = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="cuda")

        return cache_k, cache_v

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs: ...

    __call__ = nn.Module.__call__
