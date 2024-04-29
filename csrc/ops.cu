#include "fp8_gmm.h"
#include "multi_pointwise.h"
#include "multi_pad_fusion.h"

#include <torch/extension.h>

namespace fp8_gmm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gmm", &Fp8GroupedGemm, "Cutlass FP8 Grouped GEMM.");
  m.def("multi_quantize", &MultiQuantize, "Multi quantize from bf16 to fp8.");
  m.def("multi_scale_mul", &MultiScaleMul, "Multi inplace mul with 2 scales.");
  m.def("multi_pad_transpose", &MultiPadTranspose, "Multi pad and transpose.");
  m.def("multi_pad_cast_transpose", &MultiPadCastTranspose, "Multi pad, cast and transpose.");
  m.def("multi_pad_cast_transpose_dgelu", &MultiPadCastTransposeDgelu, "Multi pad, cast, transpose and dgelu.");
}

}  // namespace fp8_gmm
