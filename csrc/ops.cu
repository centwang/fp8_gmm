#include "fp8_gmm.h"

#include <torch/extension.h>

namespace fp8_gmm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gmm", &Fp8GroupedGemm, "FP8 Grouped GEMM.");
}

}  // namespace fp8_gmm
