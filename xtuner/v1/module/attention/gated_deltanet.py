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

# from ...ops.gated_deltanet.causal_conv1d import causal_conv1d_fn
# from ...ops.gated_deltanet.chunk_gated_delta_rule import chunk_gated_delta_rule
from ...ops.gated_deltanet.gen_seq_idx import gen_seq_idx
# from ...ops.gated_deltanet.rms_norm_gated import rms_norm_gated
from ..linear import build_linear
from .attn_outputs import AttnOutputs
from .causal_conv1d import causal_conv1d_triton
from xtuner.v1.utils.event_record import event_timer


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

    class FusedRMSNormGated(FLA_FusedRMSNormGated):
        def forward(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        # from .chunk_gated_delta_rule_npu.chunk_gated_delta_rule import chunk_gated_delta_rule
        # print("Loading Ascend C GDN")
        from .chunk_gated_delta_rule_npu.flash_gated_delta_rule import flash_gated_delta_rule as chunk_gated_delta_rule
        # chunk_gated_delta_rule = flash_gated_delta_rule
        # print("Using Ascend C GDN")
        
    else:
        chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

logger = get_logger()

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


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
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # # Norm before gate
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # hidden_states = weight * hidden_states.to(input_dtype)
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

        # assert causal_conv1d_fn is not None, (
        #     "causal_conv1d_fn is not available. Please install causal-conv1d to use GatedDeltaNet by `https://github.com/Dao-AILab/causal-conv1d`."
        # )
        self.causal_conv1d_fn = causal_conv1d_fn
        # assert chunk_gated_delta_rule is not None, (
        #     "chunk_gated_delta_rule is not available. Please install fla to use GatedDeltaNet by `pip install flash-linear-attention`."
        # )
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        # assert FusedRMSNormGated is not None, (
        #     "FusedRMSNormGated is not available. Please install fla to use GatedDeltaNet by `pip install flash-linear-attention`."
        # )
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

        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_rank = seq_ctx.sp_rank
        sp_size = seq_ctx.sequence_parallel_mesh.size()
        if seq_ctx.seq_idx is None:
            # SP restores the full packed sequence before convolution, so seq_idx uses the global length.
            seq_idx = gen_seq_idx(seq_len * sp_size, seq_ctx.cu_seq_lens_q)
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        A_log = A_log.to(mixed_qkv.device)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        query, key, value = torch.split(
            mixed_qkv,  # (1, L/sp_size, 8192)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
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

        query_weight, key_weight, value_weight = torch.split(
            weight,  # (8192, 4)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )
        query_weight = query_weight.chunk(sp_size, dim=0)[sp_rank]
        key_weight = key_weight.chunk(sp_size, dim=0)[sp_rank]
        value_weight = value_weight.chunk(sp_size, dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(sp_size, dim=0)[sp_rank]

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
                chunk_indices=seq_ctx.chunk_indices,
            )
            key, _ = causal_conv1d_triton(
                x=key,
                weight=key_weight,
                H=int(self.num_k_heads/sp_size),
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
                chunk_indices=seq_ctx.chunk_indices,
            )
            value, _ = causal_conv1d_triton(
                x=value,
                weight=value_weight,
                H=int(self.num_v_heads/sp_size),
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
                chunk_indices=seq_ctx.chunk_indices,
            )

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            # query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            # key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        
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
        # core_attn_out = torch.randn_like(query)
        # if seq_ctx.cu_seq_lens_q is not None:
        #     seq_ctx.cu_seq_lens_q = seq_ctx.cu_seq_lens_q.to(origin_device)
        # core_attn_out = nonvarlen_to_varlen(seq_ctx.cu_seq_lens_q, core_attn_out)
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

    def _serial_sp_initial_states(
        self,
        hidden_states: torch.Tensor,
        cache,
        num_docs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_shape = (1, self.conv_dim, self.conv_kernel_size)
        conv_history = cache.history_conv_state
        if conv_history is not None and conv_history.numel() > 0:
            first_conv = conv_history.unsqueeze(0) if conv_history.ndim == 2 else conv_history
        else:
            first_conv = hidden_states.new_zeros(conv_shape)
        conv_rest = hidden_states.new_zeros((num_docs - 1, *conv_shape[1:]))

        delta_shape = (
            1,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )
        delta_history = cache.history_delta_state
        if delta_history is not None and delta_history.numel() > 0:
            first_delta = delta_history.unsqueeze(0) if delta_history.ndim == 3 else delta_history
        else:
            first_delta = hidden_states.new_zeros(delta_shape)
        delta_rest = hidden_states.new_zeros((num_docs - 1, *delta_shape[1:]))

        return (
            torch.cat((first_conv, conv_rest), dim=0),
            torch.cat((first_delta, delta_rest), dim=0),
        )

    def _store_serial_sp_final_states(
        self,
        cache,
        final_conv_state: torch.Tensor,
        final_delta_state: torch.Tensor,
    ) -> None:
        doc_ids = cache.chunk_doc_ids[cache.chunk_idx]
        assert doc_ids, f"chunk {cache.chunk_idx} has no document"
        assert len(doc_ids) == final_conv_state.shape[0] == final_delta_state.shape[0]

        last_doc_end = cache.doc_ranges[doc_ids[-1]][1]
        chunk_end = (cache.chunk_idx + 1) * cache.chunk_size
        if last_doc_end > chunk_end:
            cache.output_conv_state = final_conv_state[-1].clone()
            cache.output_delta_state = final_delta_state[-1].clone()
        else:
            cache.output_conv_state = final_conv_state[:0].clone()
            cache.output_delta_state = final_delta_state[:0].clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
    ) -> AttnOutputs:
        cache = getattr(seq_ctx, "kvcache", None)
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            assert cache is None, "serial-SP GDN state carry currently requires sequence_parallel_size=1"
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
        if bias is not None and isinstance(bias, DTensor):
            bias = bias.to_local()

        initial_conv_state = None
        initial_delta_state = None
        if cache is not None:
            num_docs = seq_ctx.cu_seq_lens_q.numel() - 1
            assert num_docs == len(cache.chunk_doc_ids[cache.chunk_idx])
            initial_conv_state, initial_delta_state = self._serial_sp_initial_states(
                hidden_states,
                cache,
                num_docs,
            )

        if seq_ctx.seq_idx is None:
            # Keep seq_idx generation in a custom op so full-graph compile avoids Python tensor construction.
            seq_idx = gen_seq_idx(seq_len, seq_ctx.cu_seq_lens_q)
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        # TODO: due to the limitation of scatter_dim=1 in ulysses_all_to_all,
        # the implementation is very inelegant and inefficient, and needs to be refactored in the future.
        final_conv_state = None
        if cache is not None:
            mixed_qkv, final_conv_state = causal_conv1d_triton(
                x=mixed_qkv,
                weight=weight,
                H=2 * self.num_k_heads + self.num_v_heads,
                bias=bias,
                initial_state=initial_conv_state,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
                chunk_indices=seq_ctx.chunk_indices,
                output_final_state=True,
            )
        elif self.causal_conv1d_fn is not None:
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
            # new_conv = replace_conv1d(weight.unsqueeze(1), bias, self.conv1d)
            # mixed_qkv = F.silu(new_conv(mixed_qkv)[:,:,:seq_len])
            # mixed_qkv = F.silu(F.conv1d(mixed_qkv,weight.unsqueeze(1),bias,padding=weight.shape[1]-1,groups=weight.shape[0])[:,:,:seq_len])
            if seq_ctx.cu_seq_lens_q is not None and seq_ctx.cu_seq_lens_q.device != mixed_qkv.device:
                seq_ctx.cu_seq_lens_q = seq_ctx.cu_seq_lens_q.to(mixed_qkv.device)
            mixed_qkv, _ = causal_conv1d_triton(
                x=mixed_qkv,
                weight=weight,
                H=2*self.num_k_heads+self.num_v_heads,
                bias=bias,
                activation=self.activation,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
                chunk_indices=seq_ctx.chunk_indices,
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
        # query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        # key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        # value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

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
            # query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            # key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
        
        core_attn_out, final_delta_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_delta_state,
            output_final_state=cache is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_list,
            chunk_indices=seq_ctx.chunk_indices,
            chunk_indices_list=seq_ctx.chunk_indices_list 
        )

        if cache is not None:
            assert final_conv_state is not None and final_delta_state is not None
            self._store_serial_sp_final_states(
                cache,
                final_conv_state,
                final_delta_state,
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
