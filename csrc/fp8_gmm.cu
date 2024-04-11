#include "fp8_gmm.h"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace fp8_gmm {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
using ElementAF = cutlass::float_e4m3_t;  // Element type for A matrix operand for forward
using ElementAB = cutlass::float_e5m2_t;  // Element type for A matrix operand for backward
using ElementB = cutlass::float_e4m3_t;   // Element type for B matrix operand
using ElementC = cutlass::bfloat16_t;     // Element type for C and D matrix operands

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentAF =
    128 /
    cutlass::sizeof_bits<ElementAF>::value;  // Alignment of A matrix in units of elements (up to 16 bytes) for forward
constexpr int AlignmentAB =
    128 /
    cutlass::sizeof_bits<ElementAB>::value;  // Alignment of A matrix in units of elements (up to 16 bytes) for backward

// B matrix configuration
using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;              // Operator class tag
using TileShape = Shape<_256, _128, _64>;                          // Threadblock-level tile size
using ClusterShape = Shape<_2, _2, _1>;                            // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;  // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;                      // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC, LayoutC*,
    AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule>::CollectiveOp;

using CollectiveMainloopF = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementAF, LayoutA*, AlignmentAF, ElementB, LayoutB*, AlignmentB, ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using CollectiveMainloopB = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementAB, LayoutA*, AlignmentAB, ElementB, LayoutB*, AlignmentB, ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using Fp8GroupedGemmKernelF =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopF, CollectiveEpilogue>;

using Fp8GroupedGemmKernelB =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopB, CollectiveEpilogue>;

using Fp8GroupedGemmF = cutlass::gemm::device::GemmUniversalAdapter<Fp8GroupedGemmKernelF>;
using Fp8GroupedGemmB = cutlass::gemm::device::GemmUniversalAdapter<Fp8GroupedGemmKernelB>;

template <typename Gemm>
torch::Tensor CutlassFp8GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes) {
  using GemmElementA = typename Gemm::ElementA;
  using GemmElementB = typename Gemm::ElementB;
  using GemmElementC = typename Gemm::EpilogueOutputOp::ElementOutput;
  using GemmStrideA = typename Gemm::GemmKernel::UnderlyingStrideA;
  using GemmStrideB = typename Gemm::GemmKernel::UnderlyingStrideB;
  using GemmStrideC = typename Gemm::GemmKernel::UnderlyingStrideC;
  const size_t num_experts = batch_sizes.size(0);
  const int K = static_cast<int>(b.size(2)), N = static_cast<int>(b.size(1));
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  problem_sizes_host.reserve(num_experts);
  std::vector<GemmElementA*> ptr_A_host(num_experts);
  std::vector<GemmElementB*> ptr_B_host(num_experts);
  std::vector<GemmElementC*> ptr_C_host(num_experts);
  std::vector<GemmStrideA> stride_A_host;
  std::vector<GemmStrideB> stride_B_host;
  std::vector<GemmStrideC> stride_C_host;
  int64_t elements_A = 0, elements_B = 0, elements_C = 0;
  for (size_t i = 0; i < num_experts; ++i) {
    int M = static_cast<int>(batch_sizes.data_ptr<int64_t>()[i]);
    problem_sizes_host.push_back({M, N, K});
    ptr_A_host[i] = (GemmElementA*)a.data_ptr() + elements_A;
    ptr_B_host[i] = (GemmElementB*)b.data_ptr() + elements_B;
    ptr_C_host[i] = (GemmElementC*)c.data_ptr() + elements_C;
    elements_A += (M * K);
    elements_B += (K * N);
    elements_C += (M * N);
    stride_A_host.push_back(cutlass::make_cute_packed_stride(GemmStrideA{}, cute::make_shape(M, K, Int<1>{})));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(GemmStrideB{}, cute::make_shape(N, K, Int<1>{})));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(GemmStrideC{}, cute::make_shape(M, N, Int<1>{})));
  }

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<const GemmElementA*> ptr_A;
  cutlass::DeviceAllocation<const GemmElementB*> ptr_B;
  cutlass::DeviceAllocation<GemmElementC*> ptr_C;
  cutlass::DeviceAllocation<GemmStrideA> stride_A;
  cutlass::DeviceAllocation<GemmStrideB> stride_B;
  cutlass::DeviceAllocation<GemmStrideC> stride_C;
  problem_sizes.reset(num_experts);
  problem_sizes.copy_from_host(problem_sizes_host.data());
  ptr_A.reset(num_experts);
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(num_experts);
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.reset(num_experts);
  ptr_C.copy_from_host(ptr_C_host.data());
  stride_A.reset(num_experts);
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(num_experts);
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(num_experts);
  stride_C.copy_from_host(stride_C_host.data());

  typename Gemm::EpilogueOutputOp::Params params;
  params = typename Gemm::EpilogueOutputOp::Params(ElementAccumulator(1.0f), ElementAccumulator(0.0f));
  typename Gemm::Arguments arguments;
  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {static_cast<int>(num_experts), problem_sizes.get(), problem_sizes_host.data()},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {params, const_cast<const GemmElementC**>(ptr_C.get()), stride_C.get(), ptr_C.get(), stride_C.get()}};

  Gemm gemm;
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Initialize the kernel.
  if (gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
  if (gemm.run() != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }
  return c;
}

void Fp8GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes) {
  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn || a.scalar_type() == torch::kFloat8_e5m2);

  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_out, hidden_in) for 'b', i.e., trans_b = True.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn);

  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = b.size(2);
  int64_t hidden_out = b.size(1);
  TORCH_CHECK(hidden_in == a.size(1));

  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);

  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());

  if (a.scalar_type() == torch::kFloat8_e4m3fn) {
    CutlassFp8GroupedGemm<Fp8GroupedGemmF>(a, b, c, batch_sizes);
  } else {
    CutlassFp8GroupedGemm<Fp8GroupedGemmB>(a, b, c, batch_sizes);
  }
}

}  // namespace fp8_gmm
