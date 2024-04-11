#include <torch/extension.h>

namespace fp8_gmm {

void Fp8GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes);

}  // namespace fp8_gmm
