#pragma once

#include <cute/config_mpu.hpp>            // CUTE_HOST_DEVICE, CUTE_STL_NAMESPACE MAXL_STL_NS

namespace maxl
{
template <class... T>
struct type_list {};

// get<I> for type_list<T...>
//   requires tuple_element_t<I,type_list<T...>> to have std::is_default_constructible
template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
MAXL_STL_NS::tuple_element_t<I, type_list<T...>>
get(type_list<T...> const& t) noexcept {
  return {};
}

}// end namespace maxl

#if defined(__CUDACC_RTC__)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

// namespace MAXL_STL_NS
// {

// template <class... T>
// struct tuple_size<cute::type_list<T...>>
//     : MAXL_STL_NS::integral_constant<size_t, sizeof...(T)>
// {};

// template <size_t I, class... T>
// struct tuple_element<I, cute::type_list<T...>>
// {
//   using type = typename MAXL_STL_NS::tuple_element<I, MAXL_STL_NS::tuple<T...>>::type;
// };

// } // end namespace std


#if 0
namespace cute
{

template <class... T>
struct type_list {};

// get<I> for type_list<T...>
//   requires tuple_element_t<I,type_list<T...>> to have std::is_default_constructible
template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
CUTE_STL_NAMESPACE::tuple_element_t<I, type_list<T...>>
get(type_list<T...> const& t) noexcept {
  return {};
}

} // end namespace cute

//
// Specialize tuple-related functionality for cute::type_list
//

#if defined(__CUDACC_RTC__)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

namespace CUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<cute::type_list<T...>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::type_list<T...>>
{
  using type = typename CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>::type;
};

} // end namespace std

#ifdef CUTE_STL_NAMESPACE_IS_CUDA_STD
namespace std
{

#if defined(__CUDACC_RTC__)
template <class... _Tp>
struct tuple_size;

template <size_t _Ip, class... _Tp>
struct tuple_element;
#endif

template <class... T>
struct tuple_size<cute::type_list<T...>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::type_list<T...>>
{
  using type = typename CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>::type;
};

} // end namespace std
#endif // CUTE_STL_NAMESPACE_IS_CUDA_STD
#endif