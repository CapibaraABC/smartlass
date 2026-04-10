#pragma once

#include <cute/config_mpu.hpp>                     // CUTE_INLINE_CONSTANT, CUTE_HOST_DEVICE
#include <cute/container/tuple_mpu.hpp>            // cute::is_tuple
#include <cute/numeric/integral_constant_mpu.hpp>  // cute::false_type, cute::true_type

namespace maxl
{

// For slicing
struct Underscore : Int<0> {};

CUTE_INLINE_CONSTANT Underscore _;

// Convenient alias
using X = Underscore;

// Treat Underscore as an integral like integral_constant
template <>
struct is_integral<Underscore> : true_type {};

template <class T>
struct is_underscore : false_type {};
template <>
struct is_underscore<Underscore> : true_type {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct has_elem : false_type {};
template <class Elem>
struct has_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct has_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : has_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct has_elem<Tuple, Elem, seq<Is...>>
    : disjunction<has_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct all_elem : false_type {};
template <class Elem>
struct all_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct all_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : all_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct all_elem<Tuple, Elem, seq<Is...>>
    : conjunction<all_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting Underscore member
template <class Tuple>
using has_underscore = has_elem<Tuple, Underscore>;

template <class Tuple>
using all_underscore = all_elem<Tuple, Underscore>;

template <class Tuple>
using has_int1 = has_elem<Tuple, Int<1>>;

template <class Tuple>
using has_int0 = has_elem<Tuple, Int<0>>;

//
// Slice keeps only the elements of Tuple B that are paired with an Underscore
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return maxl::tuple<B>{b};
  } else {
    return maxl::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that slice(_,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return b;
  } else {
    return maxl::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Dice keeps only the elements of Tuple B that are paired with an Int
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return maxl::tuple<>{};
  } else {
    return maxl::tuple<B>{b};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that dice(1,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return maxl::tuple<>{};
  } else {
    return b;
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace maxl

#if 0
namespace cute
{

// For slicing
struct Underscore : Int<0> {};

CUTE_INLINE_CONSTANT Underscore _;

// Convenient alias
using X = Underscore;

// Treat Underscore as an integral like integral_constant
template <>
struct is_integral<Underscore> : true_type {};

template <class T>
struct is_underscore : false_type {};
template <>
struct is_underscore<Underscore> : true_type {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct has_elem : false_type {};
template <class Elem>
struct has_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct has_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : has_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct has_elem<Tuple, Elem, seq<Is...>>
    : disjunction<has_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct all_elem : false_type {};
template <class Elem>
struct all_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct all_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : all_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct all_elem<Tuple, Elem, seq<Is...>>
    : conjunction<all_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting Underscore member
template <class Tuple>
using has_underscore = has_elem<Tuple, Underscore>;

template <class Tuple>
using all_underscore = all_elem<Tuple, Underscore>;

template <class Tuple>
using has_int1 = has_elem<Tuple, Int<1>>;

template <class Tuple>
using has_int0 = has_elem<Tuple, Int<0>>;

//
// Slice keeps only the elements of Tuple B that are paired with an Underscore
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<B>{b};
  } else {
    return cute::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that slice(_,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return b;
  } else {
    return cute::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Dice keeps only the elements of Tuple B that are paired with an Int
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<>{};
  } else {
    return cute::tuple<B>{b};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that dice(1,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<>{};
  } else {
    return b;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

CUTE_HOST_DEVICE void print(Underscore const&) {
  printf("_");
}

#if !defined(__CUDACC_RTC__)
CUTE_HOST std::ostream& operator<<(std::ostream& os, Underscore const&) {
  return os << "_";
}
#endif

} // end namespace cute
#endif