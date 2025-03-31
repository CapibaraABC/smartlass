/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Barrier Operations on SM90+
*/

#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#include <cutlass/arch/memory_sm75.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100_tma.hpp> 
#include <cutlass/arch/config.h>        

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && (__CUDACC_VER_MAJOR__ >= 12)
#define CUDA_BARRIER_ENABLED 1
#else
#define CUDA_BARRIER_ENABLED 0
#endif


#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED))
#define CUTLASS_ARCH_TCGEN_ENABLED 1
#endif


namespace cutlass {
/// @brief
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////
CUTLASS_HOST_DEVICE void fence_view_async_shared();

namespace detail { // namespace detail begin

// Single threaded versions that need to be called in an elect_one region
template<typename T, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array(T ptr, int arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    ptr[i].init(arv_cnt);
  }
}

template<typename T, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array(uint64_t *ptr, int arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    T::init(&ptr[i], arv_cnt);
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_pair(FullBarrier full_barriers, EmptyBarrier empty_barriers, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    full_barriers[i].init(full_barrier_arv_cnt);
    empty_barriers[i].init(empty_barrier_arv_cnt);
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_pair(uint64_t *full_barriers_ptr, uint64_t *empty_barriers_ptr, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    FullBarrier::init(&full_barriers_ptr[i], full_barrier_arv_cnt);
    EmptyBarrier::init(&empty_barriers_ptr[i], empty_barrier_arv_cnt);
  }
}

// Aligned versions that need to be call warp wide
template<typename T, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_aligned(T ptr, int arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      ptr[i].init(arv_cnt);
    }
  }
}

template<typename T, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_aligned(uint64_t *ptr, int arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      T::init(&ptr[i], arv_cnt);
    }
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_pair_aligned(FullBarrier full_barriers, EmptyBarrier empty_barriers, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      full_barriers[i].init(full_barrier_arv_cnt);
      empty_barriers[i].init(empty_barrier_arv_cnt);
    }
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_HOST_DEVICE
void initialize_barrier_array_pair_aligned(uint64_t *full_barriers_ptr, uint64_t *empty_barriers_ptr, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      FullBarrier::init(&full_barriers_ptr[i], full_barrier_arv_cnt);
      EmptyBarrier::init(&empty_barriers_ptr[i], empty_barrier_arv_cnt);
    }
  }
}

} // namespace detail end




// There are 16 Named Barriers provided by Hardware starting in Hopper
// Their IDs are in the range 0-15
// Number of threads syncing using the barrier must be a multiple of warp-size
// ID 0 should not be used for safety, as other driver APIs (i.e. __syncthreads)
// may use it and conflict with other uses.


// Enumerates the reserved named barriers to avoid potential conflicts
// This enum class specifies the NamedBarriers reserved by CUTLASS.
enum class ReservedNamedBarriers { 
  EpilogueBarrier = 1,
  TransposeBarrier = 2,
  TransformBarrier = 3,
  StreamkBarrier0 = 4,
  StreamkBarrier1 = 5
  , TmemAllocBarrier = 6 
  , FirstUserBarrier = StreamkBarrier1 + 1
};


class NamedBarrier {

  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  // Note that should be set to the final barrier ID, including ReserveNamedBarrierCount should be considered
  uint32_t const id_;

 public:

  // Constructor for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers)
      : num_threads_(num_threads), id_(static_cast<uint32_t>(reserved_named_barriers)) {}

  // Constructor for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id + ReservedNamedBarrierCount) {
    CUTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers && "Effective barrier_id should not exceed 16.");
  }

  CUTLASS_DEVICE
  void arrive_and_wait() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_and_wait_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_and_wait_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_and_wait_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    NamedBarrier::arrive_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void sync() const {
    NamedBarrier::arrive_and_wait();
  }

  //  Static variants

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    arrive_and_wait_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_and_wait_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
    arrive_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    sync_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    sync_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }


 private:
  CUTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_and_wait_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("barrier.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }

 public:
  // Currently we reserve 8 NamedBarriers for CUTLASS' own use cases, 
  // while leaving the renaming for general users.
  static const uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(ReservedNamedBarriers::FirstUserBarrier);
  static const uint32_t HardwareMaxNumNamedBarriers = 16;

};






// ////////////////////////////////////////////////////////////////////////////////////////////////////

// // Hopper introduces a new cluster-wide barrier which handle with Cluster-wide arrive-wait behaviour.
// // This is an extension to the Ampere arrive-wait barriers
// // Note : Ampere arrive-wait Barriers have a larger max-arrive count (2^30) than Hopper arrive-wait Barriers (2^20).
// struct ClusterBarrier {

//   using ValueType = uint64_t;

// protected:
//   // Can never be initialized - can only be aliased to smem
//   ValueType barrier_;

// public:

//   CUTLASS_DEVICE
//   ClusterBarrier() = delete;

//   CUTLASS_DEVICE
//   void init(uint32_t arrive_count) const {
//     ClusterBarrier::init(&this->barrier_, arrive_count);
//   }

//   CUTLASS_DEVICE
//   bool test_wait(uint32_t phase, uint32_t pred=true) const {
//     return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
//   }

//   CUTLASS_DEVICE
//   bool try_wait(uint32_t phase) const {
//     return ClusterBarrier::try_wait(&this->barrier_, phase);
//   }

//   CUTLASS_DEVICE
//   void wait(uint32_t phase) const {
//     ClusterBarrier::wait(&this->barrier_, phase);
//   }

//   // Barrier arrive on local smem
//   CUTLASS_DEVICE
//   void arrive() const {
//     ClusterBarrier::arrive(&this->barrier_);
//   }

//   // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
//   CUTLASS_DEVICE
//   void arrive(uint32_t cta_id, uint32_t pred = true ) const {
//     ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
//   }

//   //
//   //  Static Versions
//   //
//   CUTLASS_DEVICE
//   static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.init.shared::cta.b64 [%1], %0; \n"
//         "}"
//         :
//         : "r"(arrive_count), "r"(smem_addr));
//     cutlass::arch::synclog_emit_cluster_barrier_init(__LINE__, smem_addr, arrive_count);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   // Static version of wait - in case we don't want to burn a register
//   CUTLASS_DEVICE
//   static void wait(ValueType const* smem_ptr, uint32_t phase) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     cutlass::arch::synclog_emit_cluster_barrier_wait(__LINE__, smem_addr, phase);
//     // Arbitrarily large timer value after which try-wait expires and re-tries.
//     uint32_t ticks = 0x989680;
//     asm volatile(
//         "{\n\t"
//         ".reg .pred       P1; \n\t"
//         "LAB_WAIT: \n\t"
//         "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
//         "@P1 bra DONE; \n\t"
//         "bra     LAB_WAIT; \n\t"
//         "DONE: \n\t"
//         "}"
//         :
//         : "r"(smem_addr), "r"(phase), "r"(ticks));

// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   CUTLASS_DEVICE
//   static bool test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     cutlass::arch::synclog_emit_cluster_barrier_test_wait(__LINE__, smem_addr, phase, pred);
//     uint32_t waitComplete;

//     asm volatile(
//         "{\n\t"
//         ".reg .pred P1; \n\t"
//         ".reg .pred P2; \n\t"
//         "setp.eq.u32 P2, %3, 1;\n\t"
//         "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
//         "selp.b32 %0, 1, 0, P1; \n\t"
//         "}"
//         : "=r"(waitComplete)
//         : "r"(smem_addr), "r"(phase), "r"(pred));

//     return static_cast<bool>(waitComplete);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//     return 0;
//   }

//   CUTLASS_DEVICE
//   static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     cutlass::arch::synclog_emit_cluster_barrier_try_wait(__LINE__, smem_addr, phase);
//     uint32_t waitComplete;

//     asm volatile(
//         "{\n\t"
//         ".reg .pred P1; \n\t"
//         "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
//         "selp.b32 %0, 1, 0, P1; \n\t"
//         "}"
//         : "=r"(waitComplete)
//         : "r"(smem_addr), "r"(phase));

//     return static_cast<bool>(waitComplete);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//     return 0;
//   }

//   // Static Predicated version of the above - in case we know the address.
//   CUTLASS_DEVICE
//   static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     if (pred) {
//       asm volatile(
//           "{\n\t"
//           ".reg .b32 remAddr32;\n\t"
//           "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
//           "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
//           "}"
//           :
//           : "r"(smem_addr), "r"(cta_id));
//     }

//     cutlass::arch::synclog_emit_cluster_barrier_arrive_cluster(__LINE__, smem_addr, cta_id, pred);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   // Barrier arrive on local smem
//   CUTLASS_DEVICE
//   static void arrive(ValueType const* smem_ptr) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
//         "}"
//         :
//         : "r"(smem_addr));
//     cutlass::arch::synclog_emit_cluster_barrier_arrive(__LINE__, smem_addr);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   CUTLASS_DEVICE
//   static void invalidate(ValueType const* smem_ptr) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.inval.shared::cta.b64 [%0]; \n\t"
//         "}"
//         :
//         : "r"(smem_addr));
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }
// };

// ////////////////////////////////////////////////////////////////////////////////////////////////////

// // SM90 also introduces a new type of cluster-barrier which supports sync.
// // not just based on Arrive Count, but also transaction count (in bytes)
// struct ClusterTransactionBarrier : public ClusterBarrier {

//   CUTLASS_DEVICE
//   ClusterTransactionBarrier() = delete;

//   // Performs an arrive operation + expected transaction bytes increment
//   CUTLASS_DEVICE
//   void arrive_and_expect_tx(uint32_t transaction_bytes) const {
//     ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
//   }

//   // Performs an arrive operation + expected transaction bytes increment
//   CUTLASS_DEVICE
//   void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
//     ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes , cta_id, pred);
//   }

//   // Performs an expected transaction bytes increment without doing an arrive operation
//   CUTLASS_DEVICE
//   void expect_transaction(uint32_t transaction_bytes) const {
//     ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
//   }

//   // Performs an expected transaction bytes decrement without doing an arrive operation
//   CUTLASS_DEVICE
//   void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
//     uint32_t cta_rank = cute::block_rank_in_cluster();
//     ClusterTransactionBarrier::complete_transaction(&this->barrier_, cta_rank, transaction_bytes, pred);
//   }

//   // Performs an expected transaction bytes decrement without doing an arrive operation
//   CUTLASS_DEVICE
//   void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
//     ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id, transaction_bytes, pred);
//   }

//   //
//   //  Static Versions
//   //

//   // Performs an arrive operation + expected transaction bytes increment
//   CUTLASS_DEVICE
//   static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
//         "}"
//         :
//         : "r"(transaction_bytes), "r"(smem_addr));
//     cutlass::arch::synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(__LINE__, smem_addr, transaction_bytes);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   // Performs an arrive operation + expected transaction bytes increment for a remote cta_id in a Cluster
//   CUTLASS_DEVICE
//   static void arrive_and_expect_tx(
//       ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         ".reg .pred p;\n\t"
//         ".reg .b32 remAddr32;\n\t"
//         "setp.eq.u32 p, %2, 1;\n\t"
//         "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
//         "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
//         "}"
//         :
//         : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   // Performs an expected transaction bytes increment without doing an arrive operation
//   CUTLASS_DEVICE
//   static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.expect_tx.shared::cta.b64 [%1], %0; \n\t"
//         "}"
//         :
//         : "r"(transaction_bytes), "r"(smem_addr));
//     cutlass::arch::synclog_emit_cluster_transaction_barrier_expect_transaction(__LINE__, smem_addr, transaction_bytes);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   // Performs an expected transaction bytes decrement without doing an arrive operation
//   CUTLASS_DEVICE
//   static void complete_transaction(
//       ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
// #if CUDA_BARRIER_ENABLED
//     uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
//     smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
//     asm volatile(
//         "{\n\t"
//         ".reg .pred p;\n\t"
//         "setp.eq.u32 p, %2, 1;\n\t"
//         "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64   [%1], %0;"
//         "}"
//         :
//         : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
//     cutlass::arch::synclog_emit_cluster_transaction_barrier_complete_transaction(__LINE__, smem_addr, dst_cta_id, transaction_bytes, pred);
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//   }

//   //
//   // DEPRECATED APIs
//   //
//   [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
//   void arrive_and_reset_bytes(uint32_t transaction_bytes) const {
//     arrive_and_expect_tx(transaction_bytes);
//   }
//   [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
//   void arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
//     arrive_and_expect_tx(transaction_bytes, cta_id);
//   }
//   [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE
//   void reset_bytes(uint32_t transaction_bytes) const {
//     expect_transaction(transaction_bytes);
//   }
//   [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
//   void commit(uint32_t transaction_bytes, uint32_t pred = 1) const {
//     complete_transaction(transaction_bytes, pred);
//   }
//   [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
//   void commit(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
//     complete_transaction(dst_cta_id, transaction_bytes, pred);
//   }
//   [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
//   static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
//     arrive_and_expect_tx(smem_ptr, transaction_bytes);
//   }
//   [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
//   static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
//     arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
//   }
//   [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE
//   static void reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
//     expect_transaction(smem_ptr, transaction_bytes);
//   }
//   [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
//   static void commit(ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
//     complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
//   }
// };

//txl
// 用于模拟同步的纯 C++ 版本，不涉及实际阻塞，只更新计数器并打印日志。

//=================== ClusterBarrier ===========================
struct ClusterBarrier {
public:
    using ValueType = uint64_t;

protected:
    // 计数器（此处为了模拟使用普通变量；实际多线程环境下可使用std::atomic）
    mutable ValueType barrier_;

public:
    // 默认构造函数（原始代码禁止直接构造，但这里为了测试允许构造）
    ClusterBarrier(uint32_t init_count = 0) : barrier_(init_count) {}

    // 动态成员函数

    // 初始化：设置计数器为 arrive_count
    void init(uint32_t arrive_count) const {
        barrier_ = arrive_count;
        std::cout << "[Dynamic] ClusterBarrier::init set barrier_ to " << barrier_ << std::endl;
    }

    // 测试等待：如果计数器等于 phase 则返回 true，否则返回 false
    bool test_wait(uint32_t phase, uint32_t pred = true) const {
        bool result = (pred && (barrier_ == phase));
        std::cout << "[Dynamic] ClusterBarrier::test_wait: phase = " << phase 
                  << ", barrier_ = " << barrier_ 
                  << ", result = " << (result ? "true" : "false") << std::endl;
        return result;
    }

    // 尝试等待（非阻塞）：如果计数器等于 phase 则返回 true，否则返回 false
    bool try_wait(uint32_t phase) const {
        bool result = (barrier_ == phase);
        std::cout << "[Dynamic] ClusterBarrier::try_wait: phase = " << phase 
                  << ", barrier_ = " << barrier_ 
                  << ", result = " << (result ? "true" : "false") << std::endl;
        return result;
    }

    // wait()：不阻塞，仅打印当前计数（模拟等待）
    void wait(uint32_t phase) const {
        std::cout << "[Dynamic] ClusterBarrier::wait called with phase = " << phase 
                  << ", current barrier_ = " << barrier_ << std::endl;
    }

    // arrive()：计数器加 1，并打印
    void arrive() const {
        ++barrier_;
        std::cout << "[Dynamic] ClusterBarrier::arrive, new barrier_ = " << barrier_ << std::endl;
    }

    // 带条件的到达：如果 pred 为 true，则计数器加 1，并打印 cta_id 信息
    void arrive(uint32_t cta_id, uint32_t pred = true) const {
        if (pred) {
            ++barrier_;
            std::cout << "[Dynamic] ClusterBarrier::arrive for cta_id " << cta_id 
                      << ", new barrier_ = " << barrier_ << std::endl;
        } else {
            std::cout << "[Dynamic] ClusterBarrier::arrive for cta_id " << cta_id 
                      << " skipped due to pred false" << std::endl;
        }
    }

    //=================== 静态版本 ===========================
    // 这些函数操作一个指向 ValueType 的指针

    static void init(ValueType* smem_ptr, uint32_t arrive_count) {
        *smem_ptr = arrive_count;
        std::cout << "[Static] ClusterBarrier::init set *smem_ptr to " << *smem_ptr << std::endl;
    }

    static void wait(ValueType* smem_ptr, uint32_t phase) {
        std::cout << "[Static] ClusterBarrier::wait called with phase = " << phase 
                  << ", *smem_ptr = " << *smem_ptr << std::endl;
    }

    static bool test_wait(ValueType* smem_ptr, uint32_t phase, uint32_t pred) {
        bool result = (pred && (*smem_ptr == phase));
        std::cout << "[Static] ClusterBarrier::test_wait: phase = " << phase 
                  << ", *smem_ptr = " << *smem_ptr 
                  << ", result = " << (result ? "true" : "false") << std::endl;
        return result;
    }

    static bool try_wait(ValueType* smem_ptr, uint32_t phase) {
        bool result = (*smem_ptr == phase);
        std::cout << "[Static] ClusterBarrier::try_wait: phase = " << phase 
                  << ", *smem_ptr = " << *smem_ptr 
                  << ", result = " << (result ? "true" : "false") << std::endl;
        return result;
    }

    static void arrive(ValueType* smem_ptr) {
        ++(*smem_ptr);
        std::cout << "[Static] ClusterBarrier::arrive, new *smem_ptr = " << *smem_ptr << std::endl;
    }

    static void arrive(ValueType* smem_ptr, uint32_t cta_id, uint32_t pred = true) {
        if (pred) {
            ++(*smem_ptr);
            std::cout << "[Static] ClusterBarrier::arrive for cta_id " << cta_id 
                      << ", new *smem_ptr = " << *smem_ptr << std::endl;
        } else {
            std::cout << "[Static] ClusterBarrier::arrive for cta_id " << cta_id 
                      << " skipped due to pred false" << std::endl;
        }
    }

    static void invalidate(ValueType* smem_ptr) {
        *smem_ptr = 0;
        std::cout << "[Static] ClusterBarrier::invalidate set *smem_ptr to " << *smem_ptr << std::endl;
    }
};

//=================== ClusterTransactionBarrier ===========================
// 继承自 ClusterBarrier，增加事务字节计数相关的操作
struct ClusterTransactionBarrier : public ClusterBarrier {
    // 构造函数
    ClusterTransactionBarrier(uint32_t arrive_count = 0) : ClusterBarrier(arrive_count) {}

    // 动态成员：到达并期待事务字节数增加
    void arrive_and_expect_tx(uint32_t transaction_bytes) const {
        barrier_ += transaction_bytes;
        std::cout << "[Dynamic] ClusterTransactionBarrier::arrive_and_expect_tx: added " 
                  << transaction_bytes << " bytes, new barrier_ = " << barrier_ << std::endl;
    }

    // 带条件的版本
    void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = true) const {
        if (pred) {
            barrier_ += transaction_bytes;
            std::cout << "[Dynamic] ClusterTransactionBarrier::arrive_and_expect_tx for cta_id " 
                      << cta_id << ": added " << transaction_bytes << " bytes, new barrier_ = " << barrier_ << std::endl;
        } else {
            std::cout << "[Dynamic] ClusterTransactionBarrier::arrive_and_expect_tx for cta_id " 
                      << cta_id << " skipped due to pred false" << std::endl;
        }
    }

    // 动态成员：期待事务（仅增加计数器，不到达）
    void expect_transaction(uint32_t transaction_bytes) const {
        barrier_ += transaction_bytes;
        std::cout << "[Dynamic] ClusterTransactionBarrier::expect_transaction: added " 
                  << transaction_bytes << " bytes, new barrier_ = " << barrier_ << std::endl;
    }

    // 动态成员：完成事务（减少计数器）
    void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
        if (pred) {
            barrier_ -= transaction_bytes;
            std::cout << "[Dynamic] ClusterTransactionBarrier::complete_transaction: subtracted " 
                      << transaction_bytes << " bytes, new barrier_ = " << barrier_ << std::endl;
        } else {
            std::cout << "[Dynamic] ClusterTransactionBarrier::complete_transaction skipped due to pred false" << std::endl;
        }
    }

    // 带条件的完成事务，指定目标 cta_id
    void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
        if (pred) {
            barrier_ -= transaction_bytes;
            std::cout << "[Dynamic] ClusterTransactionBarrier::complete_transaction for dst_cta_id " 
                      << dst_cta_id << ": subtracted " << transaction_bytes 
                      << " bytes, new barrier_ = " << barrier_ << std::endl;
        } else {
            std::cout << "[Dynamic] ClusterTransactionBarrier::complete_transaction for dst_cta_id " 
                      << dst_cta_id << " skipped due to pred false" << std::endl;
        }
    }

    //=================== 静态版本 ===========================
    static void arrive_and_expect_tx(ValueType* smem_ptr, uint32_t transaction_bytes) {
        *smem_ptr += transaction_bytes;
        std::cout << "[Static] ClusterTransactionBarrier::arrive_and_expect_tx: added " 
                  << transaction_bytes << " bytes, new *smem_ptr = " << *smem_ptr << std::endl;
    }

    static void arrive_and_expect_tx(ValueType* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
        if (pred) {
            *smem_ptr += transaction_bytes;
            std::cout << "[Static] ClusterTransactionBarrier::arrive_and_expect_tx for cta_id " 
                      << cta_id << ": added " << transaction_bytes << " bytes, new *smem_ptr = " << *smem_ptr << std::endl;
        } else {
            std::cout << "[Static] ClusterTransactionBarrier::arrive_and_expect_tx for cta_id " 
                      << cta_id << " skipped due to pred false" << std::endl;
        }
    }

    static void expect_transaction(ValueType* smem_ptr, uint32_t transaction_bytes) {
        *smem_ptr += transaction_bytes;
        std::cout << "[Static] ClusterTransactionBarrier::expect_transaction: added " 
                  << transaction_bytes << " bytes, new *smem_ptr = " << *smem_ptr << std::endl;
    }

    static void complete_transaction(ValueType* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
        if (pred) {
            *smem_ptr -= transaction_bytes;
            std::cout << "[Static] ClusterTransactionBarrier::complete_transaction for dst_cta_id " 
                      << dst_cta_id << ": subtracted " << transaction_bytes 
                      << " bytes, new *smem_ptr = " << *smem_ptr << std::endl;
        } else {
            std::cout << "[Static] ClusterTransactionBarrier::complete_transaction for dst_cta_id " 
                      << dst_cta_id << " skipped due to pred false" << std::endl;
        }
    }
    
    //=================== DEPRECATED APIs ===========================
    [[deprecated("Use arrive_and_expect_tx instead")]]
    void arrive_and_reset_bytes(uint32_t transaction_bytes) const {
        arrive_and_expect_tx(transaction_bytes);
    }
    [[deprecated("Use arrive_and_expect_tx instead")]]
    void arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
        arrive_and_expect_tx(transaction_bytes, cta_id);
    }
    [[deprecated("Use expect_transaction instead")]]
    void reset_bytes(uint32_t transaction_bytes) const {
        expect_transaction(transaction_bytes);
    }
    [[deprecated("Use complete_transaction instead")]]
    void commit(uint32_t transaction_bytes, uint32_t pred = 1) const {
        complete_transaction(transaction_bytes, pred);
    }
    [[deprecated("Use complete_transaction instead")]]
    void commit(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
        complete_transaction(dst_cta_id, transaction_bytes, pred);
    }
    [[deprecated("Use arrive_and_expect_tx instead")]]
    static void arrive_and_reset_bytes(ValueType* smem_ptr, uint32_t transaction_bytes) {
        arrive_and_expect_tx(smem_ptr, transaction_bytes);
    }
    [[deprecated("Use arrive_and_expect_tx instead")]]
    static void arrive_and_reset_bytes(ValueType* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
        arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
    }
    [[deprecated("Use expect_transaction instead")]]
    static void reset_bytes(ValueType* smem_ptr, uint32_t transaction_bytes) {
        expect_transaction(smem_ptr, transaction_bytes);
    }
    [[deprecated("Use complete_transaction instead")]]
    static void commit(ValueType* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
        complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
    }
};

// // 辅助函数，用于模拟 barrier 初始化的fence（仅打印日志，不阻塞）
// void fence_barrier_init() {
//     std::cout << "[Host] fence_barrier_init called." << std::endl;
// }




// // 假设最大阶段数
// #define MAX_STAGES 10

// // 屏障状态结构
// struct BarrierState {
//     uint32_t phase;          // 当前阶段，取值 0 或 1
//     uint32_t arrive_count;   // 期望的总到达计数
//     uint32_t current_count;  // 当前已到达计数
// };

// struct TransactionBarrierState {
//     uint32_t phase;          // 当前阶段，取值 0 或 1
//     uint32_t arrive_count;   // 期望的总到达计数
//     uint32_t current_count;  // 当前已到达计数
//     uint32_t expected_tx;    // 总预期交易字节
//     uint32_t completed_tx;   // 总已完成交易字节
// };

// // 全局状态数组，存储在全局内存中
// __device__ BarrierState g_empty_barrier_states[MAX_STAGES] = {{0, 0, 0}};
// __device__ TransactionBarrierState g_full_barrier_states[MAX_STAGES] = {{0, 0, 0, 0, 0}};

// // ClusterBarrier 实现
// struct ClusterBarrier {
//     using ValueType = BarrierState*; // 直接指向全局内存中的状态

// protected:
//     ValueType barrier_ptr_; // 指向全局内存中的屏障状态

// public:
//     __device__ ClusterBarrier(ValueType ptr) : barrier_ptr_(ptr) {
//         // 边界检查
//         if (barrier_ptr_ < g_empty_barrier_states || barrier_ptr_ >= g_empty_barrier_states + MAX_STAGES) {
//             printf("Block %d: Invalid barrier_ptr_ %p\n", blockIdx.x, barrier_ptr_);
//         }
//     }

//     __device__
//     void init(uint32_t arrive_count) const {
//         if (threadIdx.x == 0) {
//             atomicExch(&barrier_ptr_->arrive_count, arrive_count);
//             atomicExch(&barrier_ptr_->phase, 0);
//             atomicExch(&barrier_ptr_->current_count, 0);
//             int stage = barrier_ptr_ - g_empty_barrier_states; // 仅用于调试
//             printf("Block %d, Barrier at %p (Stage %d): Initialized with arrive_count = %u\n", 
//                    blockIdx.x, barrier_ptr_, stage, arrive_count);
//         }
//         __syncthreads();
//     }

//     __device__
//     bool test_wait(uint32_t phase, uint32_t pred = true) const {
//         if (!pred) return false;
//         return (barrier_ptr_->phase == (phase & 1) && 
//                 barrier_ptr_->current_count >= barrier_ptr_->arrive_count);
//     }

//     __device__
//     bool try_wait(uint32_t phase) const {
//         return (barrier_ptr_->phase == (phase & 1) && 
//                 barrier_ptr_->current_count >= barrier_ptr_->arrive_count);
//     }

//     __device__
//     void wait(uint32_t phase) const {
//         __shared__ uint32_t local_phase;
//         if (threadIdx.x == 0) {
//             local_phase = phase & 1;
//             uint32_t timeout = 1000000; // 超时机制
//             while (barrier_ptr_->phase != local_phase || 
//                    barrier_ptr_->current_count < barrier_ptr_->arrive_count) {
//                 if (--timeout == 0) {
//                     int stage = barrier_ptr_ - g_empty_barrier_states;
//                     printf("Block %d, Barrier at %p (Stage %d): Wait timeout, phase = %u, current_count = %u, arrive_count = %u\n",
//                            blockIdx.x, barrier_ptr_, stage, barrier_ptr_->phase, 
//                            barrier_ptr_->current_count, barrier_ptr_->arrive_count);
//                     break;
//                 }
//             }
//             if (timeout > 0) {
//                 int stage = barrier_ptr_ - g_empty_barrier_states;
//                 printf("Block %d, Barrier at %p (Stage %d): Wait completed, phase = %u\n", 
//                        blockIdx.x, barrier_ptr_, stage, local_phase);
//             }
//         }
//         __syncthreads();
//     }

//     __device__
//     void arrive() const {
//         if (threadIdx.x == 0) {
//             uint32_t old_count = atomicAdd(&barrier_ptr_->current_count, 1);
//             int stage = barrier_ptr_ - g_empty_barrier_states; // 仅用于调试
//             printf("Block %d, Barrier at %p (Stage %d): Arrived, current_count = %u\n", 
//                    blockIdx.x, barrier_ptr_, stage, old_count + 1);
//             if (old_count + 1 == barrier_ptr_->arrive_count) {
//                 atomicExch(&barrier_ptr_->phase, barrier_ptr_->phase ^ 1);
//                 atomicExch(&barrier_ptr_->current_count, 0);
//                 printf("Block %d, Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, barrier_ptr_, stage, barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }

//     __device__
//     void arrive(uint32_t cta_id, uint32_t pred = true) const {
//         if (threadIdx.x == 0 && pred && blockIdx.x == cta_id) {
//             uint32_t old_count = atomicAdd(&barrier_ptr_->current_count, 1);
//             int stage = barrier_ptr_ - g_empty_barrier_states; // 仅用于调试
//             printf("Block %d, Barrier at %p (Stage %d): Arrived (remote), current_count = %u\n", 
//                    blockIdx.x, barrier_ptr_, stage, old_count + 1);
//             if (old_count + 1 == barrier_ptr_->arrive_count) {
//                 atomicExch(&barrier_ptr_->phase, barrier_ptr_->phase ^ 1);
//                 atomicExch(&barrier_ptr_->current_count, 0);
//                 printf("Block %d, Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, barrier_ptr_, stage, barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }
// };

// // ClusterTransactionBarrier 实现
// struct ClusterTransactionBarrier : public ClusterBarrier {
//     using TxValueType = TransactionBarrierState*;

// private:
//     TxValueType tx_barrier_ptr_;

// public:
//     __device__ ClusterTransactionBarrier(TxValueType ptr) 
//         : ClusterBarrier(reinterpret_cast<ValueType>(ptr)), tx_barrier_ptr_(ptr) {
//         // 边界检查
//         if (tx_barrier_ptr_ < g_full_barrier_states || tx_barrier_ptr_ >= g_full_barrier_states + MAX_STAGES) {
//             printf("Block %d: Invalid tx_barrier_ptr_ %p\n", blockIdx.x, tx_barrier_ptr_);
//         }
//     }

//     __device__
//     void init(uint32_t arrive_count) const {
//         if (threadIdx.x == 0) {
//             atomicExch(&tx_barrier_ptr_->arrive_count, arrive_count);
//             atomicExch(&tx_barrier_ptr_->phase, 0);
//             atomicExch(&tx_barrier_ptr_->current_count, 0);
//             atomicExch(&tx_barrier_ptr_->expected_tx, 0);
//             atomicExch(&tx_barrier_ptr_->completed_tx, 0);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Initialized with arrive_count = %u\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, arrive_count);
//         }
//         __syncthreads();
//     }

//     __device__
//     void arrive_and_expect_tx(uint32_t transaction_bytes) const {
//         if (threadIdx.x == 0) {
//             uint32_t old_count = atomicAdd(&tx_barrier_ptr_->current_count, 1);
//             atomicAdd(&tx_barrier_ptr_->expected_tx, transaction_bytes);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Arrived and expect %u bytes, current_count = %u\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, transaction_bytes, old_count + 1);
//             if (old_count + 1 == tx_barrier_ptr_->arrive_count && 
//                 tx_barrier_ptr_->completed_tx >= tx_barrier_ptr_->expected_tx) {
//                 atomicExch(&tx_barrier_ptr_->phase, tx_barrier_ptr_->phase ^ 1);
//                 atomicExch(&tx_barrier_ptr_->current_count, 0);
//                 atomicExch(&tx_barrier_ptr_->expected_tx, 0);
//                 atomicExch(&tx_barrier_ptr_->completed_tx, 0);
//                 printf("Block %d, Tx Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, tx_barrier_ptr_, stage, tx_barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }

//     __device__
//     void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
//         if (threadIdx.x == 0 && pred && blockIdx.x == cta_id) {
//             uint32_t old_count = atomicAdd(&tx_barrier_ptr_->current_count, 1);
//             atomicAdd(&tx_barrier_ptr_->expected_tx, transaction_bytes);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Arrived (remote) and expect %u bytes, current_count = %u\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, transaction_bytes, old_count + 1);
//             if (old_count + 1 == tx_barrier_ptr_->arrive_count && 
//                 tx_barrier_ptr_->completed_tx >= tx_barrier_ptr_->expected_tx) {
//                 atomicExch(&tx_barrier_ptr_->phase, tx_barrier_ptr_->phase ^ 1);
//                 atomicExch(&tx_barrier_ptr_->current_count, 0);
//                 atomicExch(&tx_barrier_ptr_->expected_tx, 0);
//                 atomicExch(&tx_barrier_ptr_->completed_tx, 0);
//                 printf("Block %d, Tx Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, tx_barrier_ptr_, stage, tx_barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }

//     __device__
//     void expect_transaction(uint32_t transaction_bytes) const {
//         if (threadIdx.x == 0) {
//             atomicAdd(&tx_barrier_ptr_->expected_tx, transaction_bytes);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Expect %u bytes\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, transaction_bytes);
//         }
//         __syncthreads();
//     }

//     __device__
//     void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
//         if (threadIdx.x == 0 && pred) {
//             atomicAdd(&tx_barrier_ptr_->completed_tx, transaction_bytes);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Completed %u bytes, total completed = %u\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, transaction_bytes, tx_barrier_ptr_->completed_tx);
//             if (tx_barrier_ptr_->current_count >= tx_barrier_ptr_->arrive_count && 
//                 tx_barrier_ptr_->completed_tx >= tx_barrier_ptr_->expected_tx) {
//                 atomicExch(&tx_barrier_ptr_->phase, tx_barrier_ptr_->phase ^ 1);
//                 atomicExch(&tx_barrier_ptr_->current_count, 0);
//                 atomicExch(&tx_barrier_ptr_->expected_tx, 0);
//                 atomicExch(&tx_barrier_ptr_->completed_tx, 0);
//                 printf("Block %d, Tx Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, tx_barrier_ptr_, stage, tx_barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }

//     __device__
//     void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
//         if (threadIdx.x == 0 && pred && blockIdx.x == dst_cta_id) {
//             atomicAdd(&tx_barrier_ptr_->completed_tx, transaction_bytes);
//             int stage = tx_barrier_ptr_ - g_full_barrier_states; // 仅用于调试
//             printf("Block %d, Tx Barrier at %p (Stage %d): Completed (remote) %u bytes, total completed = %u\n", 
//                    blockIdx.x, tx_barrier_ptr_, stage, transaction_bytes, tx_barrier_ptr_->completed_tx);
//             if (tx_barrier_ptr_->current_count >= tx_barrier_ptr_->arrive_count && 
//                 tx_barrier_ptr_->completed_tx >= tx_barrier_ptr_->expected_tx) {
//                 atomicExch(&tx_barrier_ptr_->phase, tx_barrier_ptr_->phase ^ 1);
//                 atomicExch(&tx_barrier_ptr_->current_count, 0);
//                 atomicExch(&tx_barrier_ptr_->expected_tx, 0);
//                 atomicExch(&tx_barrier_ptr_->completed_tx, 0);
//                 printf("Block %d, Tx Barrier at %p (Stage %d): Phase flipped to %u\n", 
//                        blockIdx.x, tx_barrier_ptr_, stage, tx_barrier_ptr_->phase);
//             }
//         }
//         __syncthreads();
//     }
// };




// Helps with visibility of barrier init operations across warps / cta / cluster
// Available as a separate function so as to batch inits across barriers and fence once
// Note : It must be composed with an appropriate sync instruction with the right scope
// to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
CUTLASS_DEVICE
void fence_barrier_init() {
#if CUDA_BARRIER_ENABLED
  cutlass::arch::synclog_emit_fence_barrier_init(__LINE__);
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}




// Issue a shared memory fence for async operations
CUTLASS_DEVICE
void fence_view_async_shared() {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
    asm volatile (
        "{\n\t"
        "fence.proxy.async.shared::cta; \n"
        "}"
        ::);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling thread 
CUTLASS_DEVICE
void cpasync_barrier_arrive(uint64_t const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
    "{\n\t"
    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
    "}"
    :
    : "r"(smem_addr));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_addr);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling thread (noinc)
CUTLASS_DEVICE
void cpasync_barrier_arrive_noinc(uint64_t const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
    "{\n\t"
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
    "}"
    :
    : "r"(smem_addr));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_addr);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////


CUTLASS_DEVICE
void umma_arrive(uint64_t const* smem_ptr) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  if (cute::elect_one_sync()) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      :"r"(bar_intptr));
  }
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

//UMMA arrive for MMA_2x1SM
CUTLASS_DEVICE
void umma_arrive_2x1SM(uint64_t const* smem_ptr) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  if (cute::elect_one_sync()) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      :"r"(bar_intptr));
  }
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// UMMA arrive for MMA_1sm + TMA_LOAD_MULTICAST combination
CUTLASS_DEVICE
void umma_arrive_multicast(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  if(cute::elect_one_sync()) {
    asm volatile(
      "{\n\t"
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
      "}" 
      :
      :"r"(bar_intptr), "h"(cta_mask));
  }
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

// UMMA arrive for MMA_2x1SM + TMA_LOAD_MULTICAST combination
CUTLASS_DEVICE
void umma_arrive_multicast_2x1SM(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  if (cute::elect_one_sync()) {
    asm volatile(
      "{\n\t"
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
      "}" 
      :
      :"r"(bar_intptr), "h"(cta_mask));
  }
#else
  asm volatile ("brkpt;\n" ::);
#endif
}

// Temporary solution for sparse kernel.
// Will remove this when we done tightly elect_one wrap.
CUTLASS_DEVICE
void umma_arrive_multicast_no_elect(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .b16 lo, hi;\n\t"
      "mov.b32 {lo, hi}, %1;\n\t"
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], lo; \n\t"
      "}" 
      :
      :"r"(bar_intptr), "r"(uint32_t(cta_mask)));
#elif defined(__CUDA_ARCH__)
  CUTLASS_NOT_IMPLEMENTED();
#endif
}

// Temporary solution for sparse kernel.
// UMMA arrive for MMA_2x1SM + TMA_LOAD_MULTICAST combination
CUTLASS_DEVICE
void umma_arrive_multicast_2x1SM_no_elect(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .b16 lo, hi;\n\t"
      "mov.b32 {lo, hi}, %1;\n\t"
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], lo; \n\t"
      "}" 
      :
      :"r"(bar_intptr), "r"(uint32_t(cta_mask)));
#else
  CUTLASS_NOT_IMPLEMENTED();
#endif
}

// Always arrive on even SM of collaborating 2 SMs.
CUTLASS_DEVICE
void umma_arrive_2x1SM_sm0(uint64_t const* smem_ptr) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr) & cute::Sm100MmaPeerBitMask;
  asm volatile (
    "{\n\t"
    "mbarrier.arrive.shared::cluster.b64 _, [%0];\n\t"
    "}"
    :
    : "r"(bar_intptr));

#else
  asm volatile ("brkpt;\n" ::);
#endif
}

CUTE_DEVICE static void fence_view_async_tmem_load() {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  asm volatile (
    "{\n\t"
    "tcgen05.wait::ld.sync.aligned; \n"
    "}"
    ::);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}

CUTE_DEVICE static void fence_view_async_tmem_store() {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  asm volatile (
    "{\n\t"
    "tcgen05.wait::st.sync.aligned; \n"
    "}"
    ::);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace cutlass
