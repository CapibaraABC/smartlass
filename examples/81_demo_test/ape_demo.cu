#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/ma100_pipeline.hpp"
#include <cute/arch/mma_aurora_desc.hpp>
#include <cute/arch/mma_aurora_gmma.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

using MainloopPipeline = cutlass::PipelineDmaAuroraAsync<2>;
using PipelineState    = typename MainloopPipeline::PipelineState;
using PipelineParams   = typename MainloopPipeline::Params;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

  using PipelineStorage = typename MainloopPipeline::SharedStorage;
  PipelineStorage pipeline_storage;
};


template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class DmaA,
          class TB, class SmemLayoutB, class DmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT DmaA const dma_a,
            TB const* B, CUTLASS_GRID_CONSTANT DmaB const dma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = dma_a.get_dma_tensor(make_shape(M,K));                   // (M,K) DMA Tensor
  Tensor mB = dma_b.get_dma_tensor(make_shape(N,K));                   // (N,K) DMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //
  // TUTORIAL:
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
  //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
  //     with the understanding that the TMA is responsible for everything in mode-0.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
  //

  auto cta_dma_a = dma_a.get_slice(blockIdx.x);
  auto cta_dma_b = dma_b.get_slice(blockIdx.y);
  Tensor tAgA = cta_dma_a.partition_S(gA);                      // (TMA,TMA_M,TMA_K,k)
  Tensor tAsA = cta_dma_a.partition_D(sA);
  Tensor tBgB = cta_dma_b.partition_S(gB);                                                   // (TMA,TMA_N,TMA_K,k)
  Tensor tBsB = cta_dma_b.partition_D(sB); 


  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int kDmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                       CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);
  if (thread0()) {
    print("mA  : "); print(mA); print("\n");
    print("tAgA  : "); print(tAgA); print("\n");
    print("tAsA  : "); print(tAsA); print("\n");
    print("gA  : "); print(gA); print("\n");
    print("sA  : "); print(sA); print("\n");
    // print("tCrC  : "); print(tCrC); print("\n");
    
    print("---------------------\n");
  }
  
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  PipelineParams mainloop_pipeline_params;
  mainloop_pipeline_params.transaction_bytes = kDmaTransactionBytes;
  MainloopPipeline mainloop_pipeline(smem.pipeline_storage, mainloop_pipeline_params, Shape<_1,_1,_1>{});
  PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
  PipelineState mainloop_pipe_consumer_state;
  // Ensure barrier init is complete on all CTAs
  cluster_sync();
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

  // Start async loads for all pipes
  int prologue_iterations = min(K_PIPE_MAX, k_tile_count);

  // if (thread0()) {
  //   print("sA  : "); print(sA); print("\n");
  //   print("tCsA  : "); print(tCsA); print("\n");
  //   print("tCrA  : "); print(tCrA); print("\n");
  //   print("tCgC  : "); print(tCgC); print("\n");
  //   print("tCrC  : "); print(tCrC); print("\n");
    
  //   print("---------------------\n");
  // }
  CUTE_UNROLL
  for (int pipe = 0; pipe < prologue_iterations; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // Set expected Tx Bytes after each reset / init
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
      copy(dma_a.with(* tma_barrier), tAgA(_,_,_,pipe), tAsA(_,_,_,pipe));
      copy(dma_b.with(* tma_barrier), tBgB(_,_,_,pipe), tBsB(_,_,_,pipe));
      ++mainloop_pipe_producer_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Define A/B partitioning and C accumulators
  //
  // TUTORIAL:
  //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
  //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
  //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
  //     there is no need for copy(tCsA, tCrA) in the mainloop.
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
  if (thread0()) {
    print("tCsA  : "); print(tCsA); print("\n");
    print("tCsB  : "); print(tCsB); print("\n");
    print("tCgC  : "); print(tCgC); print("\n");
    print("tCrA  : "); print(tCrA); print("\n");
    print("tCrB  : "); print(tCrB); print("\n");
    print("tCrC  : "); print(tCrC); print("\n");
    
    print("---------------------\n");
  }
  //
  // PIPELINED MAIN LOOP
  //
  // TUTORIAL:
  //   Rather than interleaving the stages and instructions like in SM70 and SM80,
  //     the SM90 mainloops rely on explicit producer-consumer synchronization
  //     on the purely async instructions TMA and MMA.
  //   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
  //

  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  uint cal_loop = 0;
  if(thread0()){
    int blockLoopNum = (int)size<3>(tAgA);
    printf("==== MMA_CTA# blockShape   : (%d, %d, %d)\n", (int)size<0>(cta_tiler), (int)size<1>(cta_tiler), (int)size<2>(cta_tiler));
    printf("==== MMA_CTA# blockLoop    : (%d, %d) @_[%u]_[%u]\n", gridDim.x, gridDim.y, blockIdx.x, blockIdx.y);
    printf("==== MMA_CTA# Gmem C.coord : (%d, %d), Gmem C.Shape:(%d, %d)\n", blockIdx.x, blockIdx.y, (int)size<0>(cta_tiler), (int)size<1>(cta_tiler));
  }
  CUTE_NO_UNROLL
  while (cal_loop !=  size<3>(tAgA))
  {
    // Wait for Producer to complete
    mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);
    __syncthreads();
    int read_pipe = mainloop_pipe_consumer_state.index();
    
    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();
    mainloop_pipeline.consumer_release(mainloop_pipe_consumer_state);
    // Notify that consumption is done
    ++mainloop_pipe_consumer_state;

    if ((warp_idx == 0) && lane_predicate && k_tile_count != 0)
    {
      int pipe = mainloop_pipe_producer_state.index();
      // Wait for Consumer to complete consumption
      // Set expected Tx Bytes after each reset / init
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
      if(thread0())
      {
        printf("================================\nA matrix loading, loading shape is:");print(shape(tAgA(_,_,0,pipe)));print("\n");
      }
      copy(dma_a.with(* tma_barrier), tAgA(_,_,_,k_tile), tAsA(_,_,_,pipe));
      if(thread0())
      {
        printf("================================\nB matrix loading, loading shape is:");print(shape(tBgB(_,_,0,pipe)));print("\n");
      }
      copy(dma_b.with(* tma_barrier), tBgB(_,_,_,k_tile), tBsB(_,_,_,pipe));
      ++mainloop_pipe_producer_state;
      --k_tile_count;
      ++k_tile;
    }
    ++cal_loop;
  }

  //
  // Epilogue (unpredicated)
  //

  // axpby(alpha, tCrC, beta, tCgC);
}



// Setup params for an NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<  2>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = exchange_shape(AMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = exchange_shape(AMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bN,bK,bP));
  print("sA  : "); print(sA); print("\n");
  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(Aurora_128x128x128_F16F16F16_SS<AMMA::Major::MN,AMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  TiledCopy dmaA = make_dma_copy_A_sma(SMA_DMA_LOAD{}, mA, sA(_,_,0), cta_tiler, Shape<_1,_1,_1>{});
  TiledCopy dmaB = make_dma_copy_B_sma(SMA_DMA_LOAD{}, mB, sB(_,_,0), cta_tiler, Shape<_1,_1,_1>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  // print("dimBlock  : "); print(dimBlock); print("\n");
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                           TA, decltype(sA), decltype(dmaA),
                                           TB, decltype(sB), decltype(dmaB),
                                           TC, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dmaA,
                                                             B, dmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}



int main(int argc, char** argv)
{
    int m = 256;
    int n = 256;
    int k = 256;
    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;
    TI alpha = TI(1.0f);
    TI beta  = TI(0.0f);
    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);
    for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    int ldA = 0, ldB = 0, ldC = m;
    ldA = m;
    ldB = n;
    gemm_nt(m, n, k, alpha,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
}