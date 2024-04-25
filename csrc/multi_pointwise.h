#pragma once

#include <torch/extension.h>

namespace fp8_gmm {

// Quantize/cast multiple BFloat16 tensors to FP8 tensors.
void MultiQuantize(std::vector<at::Tensor> input_list, std::vector<at::Tensor> output_list,
                   std::vector<at::Tensor> scale_list, std::vector<at::Tensor> amax_list);

// Apply input_list[i] * scale_list_1[i] * scale_list_2[i] to output_list[i].
void MultiScaleMul(std::vector<at::Tensor> input_list, std::vector<at::Tensor> scale_list_1,
                   std::vector<at::Tensor> scale_list_2);

}  // namespace fp8_gmm
