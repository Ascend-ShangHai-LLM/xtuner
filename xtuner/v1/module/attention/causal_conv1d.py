from typing import Optional, Dict

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import torch.nn as nn

# Import the existing functions from the convolution module
from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_fwd_impl, causal_conv1d_bwd_impl

class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
        output_final_state: bool = False,
    ):
        # Save necessary tensors for backward pass
        # Avoid materializing an all-zero dht when the final state is not
        # consumed (for example, on the last serial-SP chunk).
        ctx.set_materialize_grads(False)
        # x = x.transpose(-1, -2).contiguous()
        weight = weight.transpose(-1, -2).contiguous()
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.H = H
        
        # Call the forward implementation
        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            H=H,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
            output_final_state=output_final_state,
        )
        # y = y.transpose(-1, -2).contiguous()
        # weight = weight.transpose(-1, -2).contiguous()
        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        # Retrieve saved tensors from forward pass
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        activation = ctx.activation
        cu_seqlens = ctx.cu_seqlens
        chunk_indices = ctx.chunk_indices
        H = ctx.H

        # Call the backward implementation with dht (could be None)
        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=dy,
            H=H,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
        )

        # Return gradients in the order of forward inputs
        # Note: We don't return gradients for non-tensor inputs (activation, cu_seqlens, chunk_indices, output_final_state)
        return dx, dw.transpose(0, 1).contiguous(), None, db, dr, dh0, None, None, None, None


def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Causal 1D convolution with integrated forward and backward pass.

    Args:
        x: Input tensor of shape [B, T, D]
        weight: Weight tensor of shape [W, D]
        bias: Optional bias tensor of shape [D]
        residual: Optional residual tensor of shape [B, T, D]
        initial_state: Optional initial state tensor for sequence processing
        activation: Optional activation function name
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences
        output_final_state: Whether to output the final state

    Returns:
        y: Output tensor of shape [B, T, D]
        final_state: Optional final state tensor if output_final_state is True
    """
    return CausalConv1dFunction.apply(
        x, weight, H, bias, residual, initial_state, activation, cu_seqlens, chunk_indices, output_final_state
    )
