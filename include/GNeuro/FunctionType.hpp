#pragma once
#include <string>
#include "GMath/Matrix.hpp"

namespace GNeuro {
  template<typename value_t>
  struct FunctionType {
    typedef GMath::Matrix<value_t> (*activation_t)(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName);
    typedef GMath::Matrix<value_t> (*loss_t)(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName);
  };
}
