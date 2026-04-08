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
#include <cute/atom/mma_traits.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/util/type_traits.hpp>

namespace cute {

template <class... Args>
struct MMA_Atom;

template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};

template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
{
  using MMA_Op = MMAOperation;
  using Traits = MMA_Traits<MMAOperation, Args...>;

  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;
  using LayoutAPEMNK  = typename Traits::APELayoutMNK;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return MMA_Atom<decltype(traits)>{traits};
  }

  //
  // Tensor call interfaces
  //

  // Cast, check, and call fma
  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
  {
    static_assert(DLayout::rank == 1, "Expected rank-1 D tensor");
    static_assert(ALayout::rank == 1, "Expected rank-1 A tensor");
    static_assert(BLayout::rank == 1, "Expected rank-1 B tensor");
    static_assert(CLayout::rank == 1, "Expected rank-1 C tensor");

    return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);
  }

  // Three arguments reproduces C
  template <class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout>      & C) const
  {
    return call(C, A, B, C);
  }

  //
  // make_fragment_A|B|C
  //   These functions are awkward as they expect already-partitioned tensors
  //     resulting from a previous call to partition_A|B|C
  //   The reasoning is that we can inspect the layout of the partitioned data
  //     and attempt to match it in generated fragment to promote vectorization
  //     when copying from partition to fragment.
  //

  // C also makes DMDescriptor, as A/B tensor
  template <class CTensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_C(CTensor&& ctensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<3>{});  // VMN
    CUTE_STATIC_ASSERT_V((get<0,0>(ctensor.shape()) == get<1,0>(LayoutA_TV{}.shape())));// ctensor: ((AtomM,AtomN), M, N, PIPE), LayoutA_TV{}: (_1, (AtomM, AtomK))
    CUTE_STATIC_ASSERT_V((get<0,1>(ctensor.shape()) == get<1,0>(LayoutB_TV{}.shape())));

    if constexpr (has_dereference<FrgTypeC>::value) {
      // If the intended FrgTypeA is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeC, typename remove_cvref_t<CTensor>::value_type>::value
                        
                        || (sizeof_bits_v<typename remove_cvref_t<CTensor>::value_type> == 8 &&
                            (sizeof_bits_v<ValTypeC> == 8 || sizeof_bits_v<ValTypeC> == 6 || sizeof_bits_v<ValTypeC> == 4))
                      , "Expecting ValTypeC type");
      return make_tensor<FrgTypeC>(static_cast<CTensor&&>(ctensor));
    } else {
      // Else, the intended FrgTypeC is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeC>(ctensor);
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class ATensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_A(ATensor&& atensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<3>{});  // VMK
    CUTE_STATIC_ASSERT_V(size<0>(atensor) == size<1>(LayoutA_TV{}));

    if constexpr (has_dereference<FrgTypeA>::value) {
      // If the intended FrgTypeA is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeA, typename remove_cvref_t<ATensor>::value_type>::value
                        
                        || (sizeof_bits_v<typename remove_cvref_t<ATensor>::value_type> == 8 &&
                            (sizeof_bits_v<ValTypeA> == 8 || sizeof_bits_v<ValTypeA> == 6 || sizeof_bits_v<ValTypeA> == 4))
                      , "Expecting ValTypeA type");
      return make_tensor<FrgTypeA>(static_cast<ATensor&&>(atensor));
    } else {
      // Else, the intended FrgTypeA is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeA>(atensor);
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class BTensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_fragment_B(BTensor&& btensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(btensor) >= Int<3>{});  // VNK
    CUTE_STATIC_ASSERT_V(size<0>(btensor) == size<1>(LayoutB_TV{}));

    if constexpr (has_dereference<FrgTypeB>::value) {
      // If the intended FrgTypeB is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeB, typename remove_cvref_t<BTensor>::value_type>::value
                      
                      || (sizeof_bits_v<typename remove_cvref_t<BTensor>::value_type> == 8 &&
                          (sizeof_bits_v<ValTypeB> == 8 || sizeof_bits_v<ValTypeB> == 6 || sizeof_bits_v<ValTypeB> == 4))
                      
                      , "Expecting ValTypeB type");
      return make_tensor<FrgTypeB>(static_cast<BTensor&&>(btensor));
    } else {
      // Else, the intended FrgTypeB is a value type, construct a new tensor with a fragment layout
      return make_fragment_like<FrgTypeB>(btensor);
    }

    CUTE_GCC_UNREACHABLE;
  }
};

//
// A tiling of mma atoms
//

template <class TiledMMA, class ThrCoord>
struct ThrMMA;

// @tparam MMA_Atom The MMA_Atom to use in the TiledMMA
// @tparam AtomLayoutMNK The MNK-tiling of the Atom to be performed.
// @tparam PermuationsMNK Permutations to apply to each MNK-mode before tiling for the Atom.
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;
  using AtomLayoutAPEMNK  = typename MMA_Atom::LayoutAPEMNK;

  static_assert(   rank_v<AtomLayoutMNK>  == 3,   "TiledMMA requires rank-3 AtomLayoutMNK");
  static_assert(   rank_v<PermutationMNK> == 3,   "TiledMMA requires rank-3 PermutationMNK");
  static_assert( is_tuple<PermutationMNK>::value, "TiledMMA requires independent permutations of MNK.");
  static_assert(is_static<PermutationMNK>::value, "TiledMMA requires static permutations of MNK.");
  static_assert(compatible(shape(AtomLayoutMNK{}), shape(AtomLayoutAPEMNK{})), 
                "Error: The tiling layout (AtomLayoutMNK) must match the MMA Atom's intrinsic APE layout shape.");

  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;

  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}

  CUTE_HOST_DEVICE constexpr auto
  get_thr_layout_vmnk() const {
    return thr_layout_vmnk_;
  }

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))
  // where
  //   ThrV:  The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM:  The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrN:  The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_C(CTensor&& ctensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(ctensor) >= Int<2>{});
    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(permutation_mnk<0>(),
                            permutation_mnk<1>());
    auto t_tensor = logical_divide(ctensor, t_tile);                 // (PermM,PermN)

    // Tile the tensor for the Atom
    auto c_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<1>(AtomShape_MNK{})));
    auto c_tensor = zipped_divide(t_tensor, c_tile);                 // ((AtomM,AtomN),(RestM,RestN))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto ctv = make_layout(make_shape(_1{}, make_shape(size<0>(AtomShape_MNK{}), size<1>(AtomShape_MNK{}))),
                           make_stride(Int<0>{}, make_stride(_1{}, size<0>(AtomShape_MNK{}))));  //(_1, M, N) : (_0, _1, M)
    auto tv_tensor = c_tensor.compose(ctv, _);
    // auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{},_);           // ((ThrV,FrgV),(RestM,RestN))

    // Tile the tensor for the C-threads
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<0>(AtomLayoutAPEMNK{})),
                                        make_layout(size<1>(AtomLayoutAPEMNK{}))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))

    return thr_tensor;
  }

  // Tile a tensor or a layout from shape
  //   (M,K,...)
  // to shape
  //   ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
  // where
  //   ThrV: The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM: The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrK: The threads tiled in K.      layout<3>(ThrLayoutVMNK): ThrK -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestK: The values tiled in K.
  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_A(ATensor&& atensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<2>{});
    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(permutation_mnk<0>(),
                            permutation_mnk<2>());
    auto t_tensor = logical_divide(atensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto atv = make_layout(make_shape(_1{}, make_shape(size<0>(AtomShape_MNK{}), size<2>(AtomShape_MNK{}))),
                           make_stride(Int<0>{}, make_stride(_1{}, size<0>(AtomShape_MNK{}))));//(_1, M, K):(_0, _1, _M)
    auto tv_tensor = a_tensor.compose(atv, _);
    // auto tv_tensor = a_tensor.compose(AtomLayoutA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<0>(AtomLayoutAPEMNK{})),
                                        make_layout(size<2>(AtomLayoutAPEMNK{}))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  // Tile a tensor or a layout from shape
  //   (N,K,...)
  // to shape
  //   ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...)))
  // where
  //   ThrV: The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrN: The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
  //   ThrK: The threads tiled in K.      layout<3>(ThrLayoutVMNK): ThrK -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestN: The values tiled in N.
  //   RestK: The values tiled in K.
  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_B(BTensor&& btensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(btensor) >= Int<2>{});
    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(permutation_mnk<1>(),
                            permutation_mnk<2>());
    auto t_tensor = logical_divide(btensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto b_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto b_tensor = zipped_divide(t_tensor, b_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto btv = make_layout(make_shape(_1{}, make_shape(size<1>(AtomShape_MNK{}), size<2>(AtomShape_MNK{}))),
                           make_stride(Int<0>{}, make_stride(_1{}, size<1>(AtomShape_MNK{}))));//(_1, N, K):(_0, _1, _N)
    auto tv_tensor = b_tensor.compose(btv,_);
    // auto tv_tensor = b_tensor.compose(AtomLayoutB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(AtomLayoutAPEMNK{})),
                                        make_layout(size<2>(AtomLayoutAPEMNK{}))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))

    return thr_tensor;
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_slice(ThrIdx const& thr_idx) const
  {
    auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
    return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
  }

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_thread_slice(ThrIdx const& thr_idx) const
  {
    return get_slice(thr_idx);
  }

  //
  // Utility for printing and visualization
  //

  // The permutation applied to the MNK-mode data
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  permutation_mnk() const {
    static_assert(0 <= I && I < 3);
    auto perm = get<I>(PermutationMNK{});
    return conditional_return(is_underscore<decltype(perm)>{}, size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk()), perm);
  }

  // The size of the MNK-mode
  template <int I>
  CUTE_HOST_DEVICE constexpr
  auto
  tile_size_mnk() const {
    static_assert(0 <= I && I < 3);
    return size(permutation_mnk<I>());
  }
};

template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_;

  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_C(CTensor&& ctensor) const
  {
    auto thr_tensor = make_tensor(static_cast<CTensor&&>(ctensor).data(), this->thrfrg_C(ctensor.layout()));

    auto thr_vmn = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
    return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_A(ATensor&& atensor) const
  {
    auto thr_tensor = make_tensor(static_cast<ATensor&&>(atensor).data(), this->thrfrg_A(atensor.layout()));

    auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_B(BTensor&& btensor) const
  {
    auto thr_tensor = make_tensor(static_cast<BTensor&&>(btensor).data(), this->thrfrg_B(btensor.layout()));

    auto thr_vnk = make_coord(get<0>(thr_vmnk_), make_coord(get<2>(thr_vmnk_), get<3>(thr_vmnk_)));
    return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_C(CTensor&& ctensor) const
  {
    return TiledMMA::make_fragment_C(partition_C(ctensor));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_A(ATensor&& atensor) const
  {
    return TiledMMA::make_fragment_A(partition_A(atensor));
  }

  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_B(BTensor&& btensor) const
  {
    return TiledMMA::make_fragment_B(partition_B(btensor));
  }
};

//
// These tile the MMA_Atom as a whole
//

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  auto permutation_mnk = append<3>(permutations, _);

  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}

//
// Size
//

template <int I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_size(TiledMMA<Args...> const& mma)
{
  return mma.template tile_size_mnk<I>();
}

template <class... Args>
CUTE_HOST_DEVICE constexpr
auto
tile_shape(TiledMMA<Args...> const& mma)
{
  return make_shape(tile_size<0>(mma), tile_size<1>(mma), tile_size<2>(mma));
}

// Deprecate?
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
size(TiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}

// Alias
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
thr_size(TiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}

//
// Display utilities
//

template <class... Args>
CUTE_HOST_DEVICE
void
print(MMA_Atom<MMA_Traits<Args...>> const&)
{
  using Atom = MMA_Atom<MMA_Traits<Args...>>;
  print("MMA_Atom\n");
  print("  ThrID:      "); print(typename Atom::ThrID{});      print("\n");
  print("  Shape_MNK:  "); print(typename Atom::Shape_MNK{});  print("\n");
  print("  LayoutA_TV: "); print(typename Atom::LayoutA_TV{}); print("\n");
  print("  LayoutB_TV: "); print(typename Atom::LayoutB_TV{}); print("\n");
  print("  LayoutC_TV: "); print(typename Atom::LayoutC_TV{}); print("\n");
}

template <class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE
void
print(TiledMMA<Atom, TiledThr, TiledPerm> const& mma)
{
  print("TiledMMA\n");
  print("  ThrLayoutVMNK:  "); print(mma.get_thr_layout_vmnk());  print("\n");
  print("  PermutationMNK: "); print(TiledPerm{}); print("\n");
  print(static_cast<Atom const&>(mma));
}

template <class TiledMMA, class ThrVMNK>
CUTE_HOST_DEVICE
void
print(ThrMMA<TiledMMA, ThrVMNK> const& thr_mma)
{
  print("ThrMMA\n");
  print("  Thr VMNK: "); print(thr_mma.thr_vmnk_); print("\n");
  print(static_cast<TiledMMA>(thr_mma));
}

} // namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <cute/atom/mma_traits_mpu.hpp>