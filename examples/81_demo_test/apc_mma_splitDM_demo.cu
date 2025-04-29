/**
 * in this demo, mma can split DM according to mma_atom shape.
 */
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

auto bP = Int<  2>{};  // Pipeline

template <class ElementC,
          class SmemLayoutC>  // (N,K,P)
struct SharedStorage
{
  array_aligned<ElementC, size_v<SmemLayoutC>> smem_C;
};


template <class ProblemShape, class CtaTiler,
          class SmemLayoutA,
          class SmemLayoutB,class SmemLayoutC,
          class TC, class CStride, class TiledMma>
__global__ static
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TC      * C, CStride dC, TiledMma mma)
{
  // Preconditions
  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)

  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TC, SmemLayoutC>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});
  cluster_sync();

  auto threadId = threadIdx.x + threadIdx.y * blockDim.x;
  ThrMMA thr_mma = mma.get_thread_slice_new(threadId);

  // Tensor aurora_tCsA_origin = thr_mma.aurora_partition_A_v2(sA);
  Tensor aurora_tCsA = thr_mma.aurora_partition_A_split(sA);

  // Tensor aurora_tCsB_origin = thr_mma.aurora_partition_B_v2(sB);
  Tensor aurora_tCsB = thr_mma.aurora_partition_B_split(sB);

  // Tensor aurora_tCsC_origin = thr_mma.aurora_partition_C_v3(sC);
  Tensor aurora_tCsC = thr_mma.aurora_partition_C_split(sC);

  // Allocate accumulators and clear them                      
  // Tensor aurora_tCrC_origin = thr_mma.make_fragment_C_new(aurora_tCsC_origin);  // (MMA,MMA_M,MMA_N)
  // clear(aurora_tCrC_origin);
  Tensor aurora_tCrC = thr_mma.make_fragment_C_new(aurora_tCsC);
  clear(aurora_tCrC);

  // Allocate "fragments"

  // Tensor aurora_tCrA_origin = thr_mma.make_fragment_A(aurora_tCsA_origin);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor aurora_tCrA = thr_mma.make_fragment_A(aurora_tCsA);

  // Tensor aurora_tCrB_origin = thr_mma.make_fragment_B(aurora_tCsB_origin);                         // (MMA,MMA_N,MMA_K,PIPE)
  Tensor aurora_tCrB = thr_mma.make_fragment_B(aurora_tCsB);

  if (thread0()) {

    print("aurora_tCsA  : "); print(aurora_tCsA); print("\n");                  //(((_64,_2),(_2,_64)),_16,_1,(_2)):(((_64,_1),(_131072,_0)),_4,_0,(_262144))
    // print("aurora_tCsA_origin  : "); print(aurora_tCsA_origin); print("\n");    //((_64,_64),_1,_1,(_2)):((_64,_1),_0,_0,(_262144))

    print("aurora_tCsB  : "); print(aurora_tCsB); print("\n");
    // print("aurora_tCsB_origin  : "); print(aurora_tCsB_origin); print("\n");

    print("aurora_tCsC  : "); print(aurora_tCsC); print("\n");
    // print("aurora_tCsC_origin  : "); print(aurora_tCsC_origin); print("\n");

    print("aurora_tCrA  : "); print(aurora_tCrA); print("\n");
    // print("aurora_tCrA_origin  : "); print(aurora_tCrA_origin); print("\n");

    print("aurora_tCrB  : "); print(aurora_tCrB); print("\n");
    // print("aurora_tCrB_origin  : "); print(aurora_tCrB_origin); print("\n");

    print("aurora_tCrC  : "); print(aurora_tCrC); print("\n");
    // print("aurora_tCrC_origin  : "); print(aurora_tCrC_origin); print("\n");
    
    print("---------------------\n");
  }
  //A:m2k8, B:n1k8, C:m2n1, so each pipeline repeat k*inner_m*inner_n = 8*2*1 = 16 times
  gemm(mma, aurora_tCrA(_,_,_,0), aurora_tCrB(_,_,_,0), aurora_tCrC(_,_,_,0));
  gemm(mma, aurora_tCrA(_,_,_,1), aurora_tCrB(_,_,_,1), aurora_tCrC(_,_,_,1));
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

  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static) 128*128*128
  auto bM = Int<256>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  

  // Define the smem layouts (static)

  auto sA = tile_to_dm_shape(AMMA::Layout_DM_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_dm_shape(AMMA::Layout_DM_SW128_Atom<TA>{}, make_shape(bN,bK,bP));
  // Define the MMA
  // TiledMMA tiled_mma = make_tiled_mma(Aurora_128x128x128_F16F16F16_APELayoutM2N2K1_SS<AMMA::Major::MN,AMMA::Major::MN>{});
  TiledMMA tiled_mma = make_tiled_mma(Aurora_64x64x16_F16F16F16_APELayoutM2N2K1_SS<AMMA::Major::MN,AMMA::Major::MN>{});
  auto sC = tiled_mma.tile_to_dm_shape(make_shape(bM, bN, bP));
  print("sA:  ");print(sA);print("\n");
  print("sB:  ");print(sB);print("\n");
  print("sC:  ");print(sC);print("\n");
  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TC, decltype(sC)>));
  auto ape_shape = get_ape_shape_mnk(tiled_mma);
  dim3 dimBlock(get<0>(ape_shape), get<1>(ape_shape), get<2>(ape_shape));
  print("dimBlock  : "); print(dimBlock); print("\n");
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                           decltype(sA),
                                           decltype(sB),decltype(sC),
                                           TC, decltype(dC), decltype(tiled_mma)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             C, dC, tiled_mma);
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