#include <torch/extension.h>

namespace fp8_gmm {

// Cutlass GEMM with FP8 data type.
// a is FP8E4M3 or FP8E5M2 type, row-major format, shape is (m, k), where m is the sum of all batch sizes.
// b is FP8E4M3 or FP8E5M2 type, column-major format, shape is (g, n, k), where g is the number of groups.
// c is currently hard-coded to be BFloat16, row-major format, shape is (m, n).
void Fp8GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes);

}  // namespace fp8_gmm
