#pragma once
#include <string>

namespace GNeuro {
  template <typename value_t> class Network;

  template<typename value_t>
  struct Functions {
    typedef value_t (*activation_t)(value_t _in, bool _derived, std::string &_funcName);
    typedef value_t (*optimizer_t)(value_t _learningRate, value_t _loss, value_t _previousLoss);
    typedef value_t (*loss_t)(value_t _out, value_t _expected, bool _derived, std::string &_funcName);
    typedef value_t (*mutate_t)(const Network<value_t> &_network);
  };
}
