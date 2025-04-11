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
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp" // cute::false_type
// #include "cute/arch/copy_sm100.hpp" 

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

template <class T, template <int...> class U>
struct is_kernel_tag_of : cute::false_type {};

template <template <int...> class U, int... Args>
struct is_kernel_tag_of<U<Args...>, U> : cute::true_type {};

template <class T, template <int...> class U>
constexpr bool is_kernel_tag_of_v = is_kernel_tag_of<T, U>::value;

template <class T, template <int,bool> class U>
struct is_asymmetric_dma_kernel_tag_of : cute::false_type {};

template <template <int, bool> class U, int I0, bool B0>
struct is_asymmetric_dma_kernel_tag_of<U<I0, B0>, U> : cute::true_type {};

template <class T, template <int, bool> class U>
constexpr bool is_asymmetric_dma_kernel_tag_of_v = \
                              is_asymmetric_dma_kernel_tag_of<T, U>::value;

}

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

namespace detail {

enum class KernelInputTransformType {
    FastF32,
    InterleavedComplexTF32
};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////

namespace kernel::detail {

// Has_SwapAB<T>::value will be true only if:
//   class T has member SwapAB and T::SwapAB is true
template <typename T, typename = void>
struct Has_SwapAB { static constexpr bool value = false; };

template <typename T>
struct Has_SwapAB <T, CUTE_STL_NAMESPACE::void_t<decltype(T::SwapAB)>>
{ static constexpr bool value = T::SwapAB; };

template <typename T>
static constexpr bool Has_SwapAB_v = Has_SwapAB<T>::value;

} // namespace kernel::detail

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
// struct KernelCpAsyncWarpSpecialized { };
// struct KernelCpAsyncWarpSpecializedPingpong { };
// struct KernelCpAsyncWarpSpecializedCooperative { };
// struct KernelTma { };
// struct KernelTmaWarpSpecialized { };
// struct KernelTmaWarpSpecializedPingpong { 
// };
// struct KernelTmaWarpSpecializedCooperative { 
// };

struct KernelCooperative { };

// struct KernelPtrArrayTmaWarpSpecializedCooperative { };
// struct KernelPtrArrayTmaWarpSpecializedPingpong { };

// // FP8 related policies (including Blocked Scaled Accumulation)
// template<
//   // `ScaleGranularityM`/`ScaleGranularityN` specifies scaling granularity along M/N, while zero-value
//   // `ScaleGranularityM`/`ScaleGranularityN` indicates that scaling granularity is
//   // `size<0>(TileShape_MNK{})`/`size<1>(TileShape_MNK{})` along M/N.
//   int ScaleGranularityM = 0,
//   int ScaleGranularityN = 0
// >
// struct KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum: KernelTmaWarpSpecializedCooperative { };

// // Policies to opt into mixed type GEMMs
// struct KernelTmaWarpSpecializedMixedInput : KernelTmaWarpSpecialized { };
// struct KernelTmaWarpSpecializedPingpongMixedInput : KernelTmaWarpSpecializedPingpong { };
// struct KernelTmaWarpSpecializedCooperativeMixedInput: KernelTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into
// specific collective builder dispatches)
//

// // FP8 related policies (including Fast Accumulation)
// struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
// struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
// struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
// struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };
// struct KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum : KernelPtrArrayTmaWarpSpecializedPingpong { };

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelCooperative
>
struct MainloopMa100Cooperative {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Ma100;
  using Schedule = KernelSchedule;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm
