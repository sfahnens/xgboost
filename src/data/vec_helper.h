#pragma once

namespace xgboost {
namespace data {

template <typename>
struct is_std_vector : std::false_type {};
template <typename... Args>
struct is_std_vector<std::vector<Args...>> : std::true_type {};

template <typename C>
auto vec(C& c) ->
    typename std::enable_if<is_std_vector<typename std::decay<C>::type>::value,
                            C&>::type {
  return c;
}

template <typename C>
auto vec(C& c) -> decltype(c.HostVector()) {
  return c.HostVector();
}

}  // namespace data
}  // namespace xgboost