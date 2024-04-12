#include "fp8_gmm.h"
#include "multi_pointwise.h"

#include <torch/extension.h>

namespace fp8_gmm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gmm", &Fp8GroupedGemm, "FP8 Grouped GEMM.");
  m.def("multi_quantize", &MultiQuantize, "Multi quantize from bf16 to fp8.");
  m.def("multi_scale_mul", &MultiScaleMul, "Multi inplace mul with 2 scales.");
}

}  // namespace fp8_gmm
