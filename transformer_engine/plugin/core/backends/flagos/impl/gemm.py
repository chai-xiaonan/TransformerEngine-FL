# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import flag_gems


__all__ = [
    "generic_gemm_fl",
    "te_general_grouped_gemm_fl",
]

_DTYPE_TO_TORCH = {
    0: torch.uint8,
    2: torch.int32,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def _convert_dtype(dtype: Union[int, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, int):
        return _DTYPE_TO_TORCH.get(dtype, None)
    if hasattr(dtype, "value"):
        return _DTYPE_TO_TORCH.get(dtype.value, None)
    return None


def generic_gemm_fl(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    D: Optional[torch.Tensor],
    quantizer: Any,
    output_dtype: Any,
    bias: Optional[torch.Tensor],
    bias_type: Any,
    gelu: bool,
    gelu_in: Optional[torch.Tensor],
    grad: bool,
    workspace: torch.Tensor,
    workspace_size: int,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap: Optional[Any] = None,
    comm_type: Optional[Any] = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
    alpha: float = 1.0,
    beta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    assert not gelu and gelu_in is None, "Triton-Based General Gemm do not support gelu now"
    assert quantizer is None, "Triton-Based General Gemm do not support quantization now"
    assert bias is None, "Triton-Based General Gemm do not support bias now"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    s = -1
    b = -1
    orig_A_shape = A.shape
    orig_B_shape = B.shape
    shape_a_changed = False
    shape_b_changed = False

    if A.ndim == 3:
        A = A.view(-1, A.shape[-1])
        shape_a_changed = True

    if B.ndim == 3:
        s, b, _ = B.shape
        B = B.view(-1, B.shape[-1])
        shape_b_changed = True

    A_comp = A.T if transA else A
    B_comp = B.T if transB else B

    out1 = flag_gems.mm(B_comp, A_comp)

    if shape_b_changed:
        out1 = out1.view(s, b, -1)

    torch_out_dtype = _convert_dtype(output_dtype)
    if torch_out_dtype is not None and out1.dtype != torch_out_dtype:
        out1 = out1.to(torch_out_dtype)

    bias_grad = None
    gelu_input = None
    extra_output_ret = None

    if D is not None:
        if accumulate:
            flag_gems.add_(D, out1)
        else:
            flag_gems.copy_(D, out1)
        return D, bias_grad, gelu_input, extra_output_ret
    else:
        return out1, bias_grad, gelu_input, extra_output_ret


def te_general_grouped_gemm_fl(
    B: List[torch.Tensor],
    transb: bool,
    A: List[torch.Tensor],
    transa: bool,
    D: Optional[List[torch.Tensor]],
    D_type: Any,
    m_splits: List[int],
    bias: List[torch.Tensor],
    bias_type: Any,
    single_output: bool,
    pre_gelu_out: List[torch.Tensor],
    grad: bool,
    workspace: List[torch.Tensor],
    workspaceSize: int,
    accumulate: bool,
    use_split_accumulator: bool,
    math_sm_count: int,
) -> Optional[List[torch.Tensor]]:
    if single_output and D is None:
        raise ValueError("not implemented, D should be allocated for single output case.")

    num_gemms = len(A)
    if D is None:
        D = []
        for i in range(num_gemms):
            m = A[i].shape[1] if transa else A[i].shape[0]
            n = B[i].shape[0] if transb else B[i].shape[1]
            D.append(torch.empty((m, n), dtype=D[i].dtype, device=A[0].device))

    def gelu_backward(grad_output, x):
        # Approximation of GELU derivative commonly used in Transformer Engine
        cdf = 0.5 * (1.0 + flag_gems.erf(x / flag_gems.sqrt(2.0)))
        pdf = flag_gems.exp(-0.5 * x * x) / flag_gems.sqrt(2.0 * math.pi)
        return flag_gems.mul(grad_output, (cdf + x * pdf))

    temp_D = []
    for i in range(num_gemms):
        # Handle the special case of zero-element inputs
        if A[i].numel() == 0 or B[i].numel() == 0:
            if not single_output:
                if D[i].numel() != 0 and not accumulate:
                    D[i].zero_()
            else:
                out = torch.zeros(A[i].shape[0], B[i].shape[1])
            if grad and len(bias) > i and bias[i] is not None and bias[i].numel() != 0:
                bias[i].zero_()
            if (
                len(pre_gelu_out) > i
                and pre_gelu_out[i] is not None
                and pre_gelu_out[i].numel() != 0
            ):
                pre_gelu_out[i].zero_()
            continue
        a = A[i].t() if transa else A[i]
        b = B[i].t() if transb else B[i]
        # Determine presence of epilogue tensors
        has_bias = len(bias) > i and bias[i] is not None and bias[i].numel() > 0
        has_pre_gelu = (
            len(pre_gelu_out) > i and pre_gelu_out[i] is not None and pre_gelu_out[i].numel() > 0
        )

        # Forward Pass calculation
        if not grad:
            if has_bias:
                # Fused matrix multiplication and bias addition
                out = flag_gems.addmm(bias[i], a, b)
            else:
                out = flag_gems.mm(a, b)

            # Apply GELU epilogue if pre_gelu_out is provided
            if has_pre_gelu:
                pre_gelu_out[i].copy_(out)
                out = flag_gems.gelu(out)
        else:
            out = flag_gems.mm(a, b)
            if has_pre_gelu:
                out = gelu_backward(out, pre_gelu_out[i])

            # Compute bias gradients if requested
            if has_bias:
                bias_grad = out.sum(dim=0)
                if accumulate:
                    bias[i].add_(bias_grad)
                else:
                    bias[i].copy_(bias_grad)

        if not single_output:
            # Store output
            if accumulate:
                D[i].add_(out.to(D[i].dtype))
            else:
                D[i].copy_(out.to(D[i].dtype))
        else:
            temp_D.append(out.to(D[0].dtype))

    if single_output:
        if temp_D:
            temp = torch.cat(temp_D, dim=0)
            D[0].copy_(temp)

    return bias
