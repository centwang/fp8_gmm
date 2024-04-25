#pragma once

#include <torch/extension.h>

namespace fp8_gmm {

// Cast and transpose multiple BFloat16 tensors to FP8 tensors.
// For each input tensor in input_list, cast it to FP8 tensot to output_list, and transpose it to transpose_list.
// For input tensor with shape (m, n), output tensor has same shape, and transpose tensor
// has shape (n, M), where M >= m. If M > m, the extra part will be kept unchanged.
void MultiPaddedCastTranspose(std::vector<at::Tensor> input_list, std::vector<at::Tensor> output_list,
                              std::vector<at::Tensor> transpose_list, std::vector<at::Tensor> scale_list,
                              std::vector<at::Tensor> amax_list);

}  // namespace fp8_gmm
