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

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#    define CUTE_ARCH_MMA_SM90_ENABLED
#    define CUTE_ARCH_MMA_F64_SM90_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/arch/mma_sm90_desc.hpp>
// #include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_traits_mpu.hpp>
#include <cute/arch/mma_mpu_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cute/layout.hpp>                     // cute::size
#include <cute/numeric/integral_constant.hpp>  // cute::is_static
#include <cute/numeric/numeric_types.hpp>      // cute::half_t, cute::float_e4m3_t, cute::tfloat32_t, etc
#include <cute/util/type_traits.hpp>           // cute::is_same_v

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, 
          MPU::GMMA::ScaleIn scaleA = MPU::GMMA::ScaleIn::One, 
          MPU::GMMA::ScaleIn scaleB = MPU::GMMA::ScaleIn::One>
struct MPU_128x128x64_F32F32F32_4x1_SS;

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, 
          MPU::GMMA::ScaleIn scaleA = MPU::GMMA::ScaleIn::One, 
          MPU::GMMA::ScaleIn scaleB = MPU::GMMA::ScaleIn::One>
struct MPU_64x64x16_F32F32F32_4x1_SS;

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, 
          MPU::GMMA::ScaleIn scaleA = MPU::GMMA::ScaleIn::One, 
          MPU::GMMA::ScaleIn scaleB = MPU::GMMA::ScaleIn::One>
struct MPU_64x64x16_F32F32F32_1x4_SS;

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, 
          MPU::GMMA::ScaleIn scaleA = MPU::GMMA::ScaleIn::One, 
          MPU::GMMA::ScaleIn scaleB = MPU::GMMA::ScaleIn::One>
struct MPU_64x64x16_F32F32F32_2x2_SS;

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, 
          MPU::GMMA::ScaleIn scaleA = MPU::GMMA::ScaleIn::One, 
          MPU::GMMA::ScaleIn scaleB = MPU::GMMA::ScaleIn::One>
struct MPU_64x64x16_F16F16F16_4x1_SS;

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  MPU::GMMA::Major MajorA = MPU::GMMA::Major::K,
  MPU::GMMA::Major MajorB = MPU::GMMA::Major::K,
  MPU::GMMA::ScaleIn ScaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn ScaleB = MPU::GMMA::ScaleIn::One
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");

  constexpr int tileM = size<0>(TileShape_MNK{});
  constexpr int tileN = size<1>(TileShape_MNK{});
  constexpr int tileK = size<2>(TileShape_MNK{});

  if(thread0()) {
    print("tileM:"); print(tileM); print("\n");
    print("tileN:"); print(tileN); print("\n");
    print("tileK:"); print(tileK); print("\n");
  }
  if constexpr (is_same_v<ElementC, float>) {
    if constexpr (tileM % 128 == 0 && tileN % 128 == 0 && tileK % 64 == 0) {
      return MPU_128x128x64_F32F32F32_4x1_SS<MajorA, MajorB, ScaleA, ScaleB>{};
    } 
    
    else if constexpr (tileM % 64 == 0 && tileN % 64 == 0 && tileK % 16 == 0) {
      if constexpr (tileM >= tileN * 2) {
        return MPU_64x64x16_F32F32F32_4x1_SS<MajorA, MajorB, ScaleA, ScaleB>{};
      } else if constexpr (tileN >= tileM * 2) {
        return MPU_64x64x16_F32F32F32_1x4_SS<MajorA, MajorB, ScaleA, ScaleB>{};
      } else {
        return MPU_64x64x16_F32F32F32_2x2_SS<MajorA, MajorB, ScaleA, ScaleB>{};
      }
    }

  } else if constexpr (is_same_v<ElementC, half_t>) {
    if constexpr (tileM % 64 == 0 && tileN % 64 == 0 && tileK % 16 == 0) {
      return MPU_64x64x16_F16F16F16_4x1_SS<MajorA, MajorB, ScaleA, ScaleB>{};
    }
  } else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

} // end namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////
