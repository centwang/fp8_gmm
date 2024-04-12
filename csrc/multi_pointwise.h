#include <torch/extension.h>

namespace fp8_gmm {

void MultiQuantize(std::vector<at::Tensor> input_list, std::vector<at::Tensor> output_list,
               std::vector<at::Tensor> scale_list, std::vector<at::Tensor> amax_list);

void MultiScaleMul(std::vector<at::Tensor> input_list, std::vector<at::Tensor> scale_list_1,
              std::vector<at::Tensor> scale_list_2);

}  // namespace fp8_gmm
