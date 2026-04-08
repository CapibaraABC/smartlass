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

// #include <cute/arch/mma.hpp>
#include <cute/arch/mma_mpu.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#    define CUTE_ARCH_MMA_SM90A_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMMA Descriptor and utilities

// GMMA enums and utilities
namespace MPU::GMMA {

enum class LayoutType : uint8_t {
  INTERLEAVE = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

CUTE_HOST_DEVICE char const* to_string(LayoutType const& t) {
  switch (t) {
    case LayoutType::INTERLEAVE: return "INTERLEAVE";
    case LayoutType::B128:       return "B128";
    case LayoutType::B64:        return "B64";
    case LayoutType::B32:        return "B32";
  }
  return nullptr;
}

#if !defined(__CUDACC_RTC__)
// Output operator for all enums in this namespace
CUTE_HOST std::ostream& operator<<(std::ostream& os, LayoutType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace MPU::GMMA

struct DMDescriptor
{
  uint32_t shape0;    //M for matrixA, N for matrixB, M for matrixC
  uint32_t shape1;    //K for matrixA, K for matrixB, N for matrixC

  uint32_t dmAddr;    //start addr for DM
  uint32_t strideByteOffset;  //stride offset in bytes between to k-tile mma_atom

  CUTE_HOST_DEVICE constexpr
  DMDescriptor() noexcept : shape0(0), shape1(1), dmAddr(0x0), strideByteOffset(0) {}
  CUTE_HOST_DEVICE constexpr
  DMDescriptor(uint32_t s0, uint32_t s1, uint32_t dm, uint32_t stride = 0) noexcept : shape0(s0), shape1(s1), dmAddr(dm), strideByteOffset(stride) {}
  CUTE_HOST_DEVICE constexpr
  DMDescriptor(DMDescriptor const& t) noexcept : shape0(t.shape0), shape1(t.shape1), dmAddr(t.dmAddr), strideByteOffset(t.strideByteOffset) {}
  CUTE_HOST_DEVICE constexpr
  DMDescriptor(DMDescriptor && t) noexcept : shape0(t.shape0), shape1(t.shape1), dmAddr(t.dmAddr), strideByteOffset(t.strideByteOffset) {}

  CUTE_HOST_DEVICE constexpr
  DMDescriptor& operator=(DMDescriptor const& t) noexcept {
    shape0 = t.shape0;
    shape1 = t.shape1;
    dmAddr = t.dmAddr;
    strideByteOffset = t.strideByteOffset;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  DMDescriptor& operator=(DMDescriptor && t) noexcept {
    shape0 = t.shape0;
    shape1 = t.shape1;
    dmAddr = t.dmAddr;
    strideByteOffset = t.strideByteOffset;
    return *this;
  }

  // Decay to a uint32_t
  CUTE_HOST_DEVICE constexpr
  operator uint32_t() const noexcept { return dmAddr; }
};

CUTE_HOST_DEVICE constexpr
DMDescriptor operator+(DMDescriptor const& lhs, uint32_t rhs) noexcept {
  return DMDescriptor(lhs.shape0, lhs.shape1, lhs.dmAddr + rhs, lhs.strideByteOffset);
}

// Printer
CUTE_HOST_DEVICE void
print(DMDescriptor const& t)
{
#if !defined(__CUDACC_RTC__)
  printf("DMDescriptor: \n");
  printf("  shape0 :  %u\n",      t.shape0);
  printf("  shape1 :  %u\n",      t.shape1);
  printf("  dmAddr :  0x%04x\n",  t.dmAddr);
  printf("  strideByteOffset:  0x%01x\n",      t.strideByteOffset);
#endif // !defined(__CUDACC_RTC__)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////
