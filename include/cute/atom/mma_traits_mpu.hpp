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

#include <cute/pointer_flagged.hpp>            // cute::smem_ptr_flag
#include <cute/pointer_sparse.hpp>             // cute::smem_sparse_ptr_flag
#include <cute/swizzle.hpp>                    // cute::Swizzle
#include <cute/tensor_impl.hpp>                // cute::Tensor
#include <cute/arch/mma_mpu_desc.hpp>         // cute::LayoutType
#include <cute/arch/mma_mpu_gmma.hpp>            // MPU_64x64x16_F16F16F16_SS, etc
#include <cute/atom/mma_traits.hpp>            // cute::MMA_Traits
#include <cute/layout_composed.hpp>            // cute::ComposedLayout
#include <cute/numeric/integral_constant.hpp>  // cute::is_static

namespace cute {

// Fence between the async destination accumulators of GMMA & source for their dependent use
// template <class Engine, class Layout>
// CUTE_HOST_DEVICE
// void
// warpgroup_fence_operand(Tensor<Engine, Layout>& frg) {
//   CUTE_STATIC_ASSERT(is_static<Layout>::value);
//   if constexpr (is_same_v<typename Engine::value_type, float>) {
//     auto f32_frg = recast<float>(frg);
//     CUTE_UNROLL
//     for (int i = 0; i < size(f32_frg); ++i) {
//       warpgroup_fence_operand(f32_frg(i));
//     }
//   }
//   else {
//     CUTE_STATIC_ASSERT(is_rmem<Engine>::value);
//     auto u32_frg = recast<uint32_t>(frg);
//     CUTE_UNROLL
//     for (int i = 0; i < size(u32_frg); ++i) {
//       warpgroup_fence_operand(u32_frg(i));
//     }
//   }
// }

namespace MPU::GMMA {

///////////////////////////////////////////
// Common layouts for GMMA Shared Memory //
///////////////////////////////////////////

// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

// M|N-major layouts in units of Type
template <class Type>
using Layout_MN_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_INTER_Atom_Bits{}));
template <class Type>
using Layout_MN_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW32_Atom_Bits{}));
template <class Type>
using Layout_MN_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW64_Atom_Bits{}));
template <class Type>
using Layout_MN_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));

// K-major layouts in units of Type
template <class Type>
using Layout_K_INTER_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_INTER_Atom_Bits{}));
template <class Type>
using Layout_K_SW32_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW32_Atom_Bits{}));
template <class Type>
using Layout_K_SW64_Atom  = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW64_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));

// With MPU::GMMA::Major param
template <class Type, Major tnsp>
using Layout_INTER_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_INTER_Atom<Type>,
                                               Layout_K_INTER_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW32_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW32_Atom<Type>,
                                              Layout_K_SW32_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW64_Atom = typename conditional<tnsp == Major::MN,
                                              Layout_MN_SW64_Atom<Type>,
                                              Layout_K_SW64_Atom<Type>>::type;
template <class Type, Major tnsp>
using Layout_SW128_Atom = typename conditional<tnsp == Major::MN,
                                               Layout_MN_SW128_Atom<Type>,
                                               Layout_K_SW128_Atom<Type>>::type;

///////////////////////////////////////////////////////////////////////////////
// Construction method for GMMA Descriptors
///////////////////////////////////////////////////////////////////////////////

/**
* ///////////////////////////////
* // make_dm_desc<Major::MN> //
* ///////////////////////////////
*/
template <Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
DMDescriptor
make_dm_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "GMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "GMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;

  // cast smem ptr to common ptr
  auto raw_ptr = &tensor(0, 0);
  uint32_t showPtr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(raw_ptr));

  auto stride = tensor.layout().stride();

  // Result
  DMDescriptor desc;
  desc.shape0 = size<0>(layout(tensor));
  desc.shape1 = size<1>(layout(tensor));
  desc.dmAddr = showPtr;
  
  // each value holds elementByte bytes
  constexpr uint32_t elementByte = sizeof(value_type);
  desc.elementByte = elementByte;

  if constexpr (MajorMode == Major::MN) { //column major
    desc.strideByteOffset = get<1>(stride) * elementByte;
  } else if constexpr (MajorMode == Major::K) { //row major
    desc.strideByteOffset = get<0>(stride) * elementByte;
  } else {
    static_assert(MajorMode != Major::MN && MajorMode != Major::K, "Unrecognized MajorMode!");
  }

  return desc;
}

///////////////////////////////////////////////////////////////////////////////
// Higher level GMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

struct DMDescriptorIterator
{
  using reference    = DMDescriptor;
  using element_type = DMDescriptor;
  using value_type   = DMDescriptor;

  DMDescriptor dmDesc;

  // Dereference returns the DMDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return dmDesc; }

  // Advance and return a new DMDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DMDescriptorIterator operator+(Index const& i) const
  {
    // each index i corresponds to a k-tile stride
    uint32_t addrByteOffset = static_cast<uint32_t>(i) * dmDesc.elementByte;//传入的i单位是element
    return { dmDesc + addrByteOffset };
  }
};

template <class T>
CUTE_HOST_DEVICE constexpr
DMDescriptor
raw_pointer_cast(DMDescriptorIterator const& ptr) {
  return ptr.dmDesc;
}

// Recast a DMDescriptorIterator Tensor to uint64_t, it's RegType in mma_unpack
template <class NewT>
CUTE_HOST_DEVICE constexpr
DMDescriptorIterator
recast_ptr(DMDescriptorIterator const& iter) {
  static_assert(is_same<NewT, uint32_t>::value, "Can only cast DMDMDescriptorIterator to uint32_t.");
  return iter;  // Do nothing, it will still dereference to DMDescriptor and decay to uint64_t
}

CUTE_HOST_DEVICE void
print(DMDescriptorIterator) {
  printf("GMMA::DMDescriptorIterator");
}

// The GMMA Traits below have custom fragment type flags for their smem desc tensors.
// These flags specialize a MakeTensor customization point to correctly make the fragment that is desired.
template <Major>
struct smem_desc : DMDescriptorIterator {
  // cast from base DMDescriptorIterator
  CUTE_HOST_DEVICE constexpr 
  smem_desc(DMDescriptorIterator const& iter) : DMDescriptorIterator(iter) {}
  
  // construct by DMDescriptor
  CUTE_HOST_DEVICE constexpr 
  smem_desc(DMDescriptor const& desc) : DMDescriptorIterator{desc} {}
};

} // end namespace MPU::GMMA

// Customization point for creating a GMMA::smem_desc Tensor
template <MPU::GMMA::Major MajorMode>
struct MakeTensor<MPU::GMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    // return make_tensor(MPU::GMMA::DMDescriptorIterator{MPU::GMMA::make_dm_desc<MajorMode>(tensor<0>(smem_tensor))},
    //                    replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
    return make_tensor(MPU::GMMA::DMDescriptorIterator{MPU::GMMA::make_dm_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(smem_tensor.layout(), Layout<_1,_0>{}));
  }
};


///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace MPU::GMMA {

//
// Specialized mma_unpack implementation for SM90 GMMA instructions
//

template <class MMA_Op, class... MMA_Args,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<MMA_Op, MMA_Args...> const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  // SM90 GMMA take three arguments rather than four, try to assert C and D are aliased
  static_assert(is_same<typename TD::value_type, typename TC::value_type>::value, "GMMA C and D value_type must match.");
  static_assert(is_same<DLayout, CLayout>::value, "GMMA C and D layouts must match.");
  assert((void*)&C == (void*)&D);

  uint32_t m = A.data().dmDesc.shape0;
  uint32_t n = B.data().dmDesc.shape0;
  uint32_t k = A.data().dmDesc.shape1;
  uint16_t alpha = 0;

  assert(m == get<0>(shape(C)));       //A.M = C.M
  assert(n == get<1>(shape(C)));       //A.N = C.N
  assert(k == B.data().dmDesc.shape1); //A.K = B.K

  uint64_t addressA = A.data().dmDesc.dmAddr;
  uint64_t addressB = B.data().dmDesc.dmAddr;
  uint64_t addressC = C.data().dmDesc.dmAddr;
  
  auto matrixA = reinterpret_cast<void*>(addressA);
  auto matrixB = reinterpret_cast<void*>(addressB);
  auto matrixC = reinterpret_cast<void*>(addressC);

  //call APE calculate
  MMA_Op::gemmKernel(alpha, m, n, k, matrixA, matrixB, matrixC);
}


// Accumulator layouts
template<int N>
using CLayout_64xN   = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
                              Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;

using CLayout_64x8   = CLayout_64xN<  8>;
using CLayout_64x16  = CLayout_64xN< 16>;
using CLayout_64x32  = CLayout_64xN< 32>;
using CLayout_64x64  = CLayout_64xN< 64>;
using CLayout_64x96  = CLayout_64xN< 96>;
using CLayout_64x128 = CLayout_64xN<128>;
using CLayout_64x192 = CLayout_64xN<192>;
using CLayout_64x256 = CLayout_64xN<256>;

// Shared memory source layouts for any value type
template <int M, int K>
using ABLayout       = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                              Stride<  _0,Stride<    _1,Int<M>>>>;

} // end namespace MPU::GMMA

using namespace MPU;

template <
  MPU::GMMA::Major tnspA,
  MPU::GMMA::Major tnspB,
  MPU::GMMA::ScaleIn  scaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn  scaleB = MPU::GMMA::ScaleIn::One
>
struct MPU_64x64x16_F16F16F16_4x1_SS : public MPU::GMMA::MMA_64x64x16_F32F32F32_SS<
  tnspA, tnspB, scaleA, scaleB, MPU_64x64x16_F16F16F16_4x1_SS<tnspA, tnspB, scaleA, scaleB>> {};

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, MPU::GMMA::ScaleIn scaleA, MPU::GMMA::ScaleIn scaleB>
struct MMA_Traits<MPU_64x64x16_F16F16F16_4x1_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = MPU::GMMA::smem_desc<tnspA>;
  using FrgTypeB = MPU::GMMA::smem_desc<tnspB>;
  using FrgTypeC = MPU::GMMA::smem_desc<tnspA>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_1>;
  using ALayout = MPU::GMMA::ABLayout< 64, 16>;
  using BLayout = MPU::GMMA::ABLayout< 64, 16>;
  using CLayout = MPU::GMMA::CLayout_64x64;
  using APELayoutMNK = Layout<Shape<_4, _1, _1>>;    //APE layout

  MPU::GMMA::ScaleOut accumulate_ = MPU::GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////
template <
  MPU::GMMA::Major tnspA,
  MPU::GMMA::Major tnspB,
  MPU::GMMA::ScaleIn  scaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn  scaleB = MPU::GMMA::ScaleIn::One
>
struct MPU_64x64x16_F32F32F32_4x1_SS : public MPU::GMMA::MMA_64x64x16_F32F32F32_SS<
      tnspA, tnspB, scaleA, scaleB, MPU_64x64x16_F32F32F32_4x1_SS<tnspA, tnspB, scaleA, scaleB>> {};

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, MPU::GMMA::ScaleIn scaleA, MPU::GMMA::ScaleIn scaleB>
struct MMA_Traits<MPU_64x64x16_F32F32F32_4x1_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using FrgTypeA = MPU::GMMA::smem_desc<tnspA>;
  using FrgTypeB = MPU::GMMA::smem_desc<tnspB>;
  using FrgTypeC = MPU::GMMA::smem_desc<tnspA>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_1>;
  using ALayout = MPU::GMMA::ABLayout< 64, 16>;
  using BLayout = MPU::GMMA::ABLayout< 64, 16>;
  using CLayout = MPU::GMMA::CLayout_64x64;
  using APELayoutMNK = Layout<Shape<_4, _1, _1>>;    //APE layout

  MPU::GMMA::ScaleOut accumulate_ = MPU::GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////

template <
  MPU::GMMA::Major tnspA,
  MPU::GMMA::Major tnspB,
  MPU::GMMA::ScaleIn  scaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn  scaleB = MPU::GMMA::ScaleIn::One
>
struct MPU_64x64x16_F32F32F32_1x4_SS : public MPU::GMMA::MMA_64x64x16_F32F32F32_SS<
      tnspA, tnspB, scaleA, scaleB, MPU_64x64x16_F32F32F32_1x4_SS<tnspA, tnspB, scaleA, scaleB>> {};

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, MPU::GMMA::ScaleIn scaleA, MPU::GMMA::ScaleIn scaleB>
struct MMA_Traits<MPU_64x64x16_F32F32F32_1x4_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using FrgTypeA = MPU::GMMA::smem_desc<tnspA>;
  using FrgTypeB = MPU::GMMA::smem_desc<tnspB>;
  using FrgTypeC = MPU::GMMA::smem_desc<tnspA>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_1>;
  using ALayout = MPU::GMMA::ABLayout< 64, 16>;
  using BLayout = MPU::GMMA::ABLayout< 64, 16>;
  using CLayout = MPU::GMMA::CLayout_64x64;
  using APELayoutMNK = Layout<Shape<_1, _4, _1>>;    //APE layout

  MPU::GMMA::ScaleOut accumulate_ = MPU::GMMA::ScaleOut::One;
};

///////////////////////////////////////////////////////////////////////

template <
  MPU::GMMA::Major tnspA,
  MPU::GMMA::Major tnspB,
  MPU::GMMA::ScaleIn  scaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn  scaleB = MPU::GMMA::ScaleIn::One
>
struct MPU_64x64x16_F32F32F32_2x2_SS : public MPU::GMMA::MMA_64x64x16_F32F32F32_SS<
      tnspA, tnspB, scaleA, scaleB, MPU_64x64x16_F32F32F32_2x2_SS<tnspA, tnspB, scaleA, scaleB>> {};

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, MPU::GMMA::ScaleIn scaleA, MPU::GMMA::ScaleIn scaleB>
struct MMA_Traits<MPU_64x64x16_F32F32F32_2x2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using FrgTypeA = MPU::GMMA::smem_desc<tnspA>;
  using FrgTypeB = MPU::GMMA::smem_desc<tnspB>;
  using FrgTypeC = MPU::GMMA::smem_desc<tnspA>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_1>;
  using ALayout = MPU::GMMA::ABLayout< 64, 16>;
  using BLayout = MPU::GMMA::ABLayout< 64, 16>;
  using CLayout = MPU::GMMA::CLayout_64x64;
  using APELayoutMNK = Layout<Shape<_2, _2, _1>>;    //APE layout

  MPU::GMMA::ScaleOut accumulate_ = MPU::GMMA::ScaleOut::One;
};

///////////////////////////////////////////////////////////////////////

template <
  MPU::GMMA::Major tnspA,
  MPU::GMMA::Major tnspB,
  MPU::GMMA::ScaleIn  scaleA = MPU::GMMA::ScaleIn::One,
  MPU::GMMA::ScaleIn  scaleB = MPU::GMMA::ScaleIn::One
>
struct MPU_128x128x64_F32F32F32_4x1_SS : public MPU::GMMA::MMA_64x64x16_F32F32F32_SS<
      tnspA, tnspB, scaleA, scaleB, MPU_128x128x64_F32F32F32_4x1_SS<tnspA, tnspB, scaleA, scaleB>> {};

template <MPU::GMMA::Major tnspA, MPU::GMMA::Major tnspB, MPU::GMMA::ScaleIn scaleA, MPU::GMMA::ScaleIn scaleB>
struct MMA_Traits<MPU_128x128x64_F32F32F32_4x1_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using FrgTypeA = MPU::GMMA::smem_desc<tnspA>;
  using FrgTypeB = MPU::GMMA::smem_desc<tnspB>;
  using FrgTypeC = MPU::GMMA::smem_desc<tnspA>;

  using Shape_MNK = Shape<_128,_128,_64>;
  using ThrID   = Layout<_1>;
  using ALayout = MPU::GMMA::ABLayout< 128, 64>;
  using BLayout = MPU::GMMA::ABLayout< 128, 64>;
  using CLayout = MPU::GMMA::CLayout_64x64;
  using APELayoutMNK = Layout<Shape<_4, _1, _1>>;    //APE layout

  MPU::GMMA::ScaleOut accumulate_ = MPU::GMMA::ScaleOut::One;
};

} // end namespace cute