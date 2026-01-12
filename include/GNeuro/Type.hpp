#pragma once
#include <cmath>

namespace GNeuro {
  typedef double_t DECIMAL_T;
  typedef DECIMAL_T (*ACTIVATION_T)(DECIMAL_T _in, bool _derived);
  typedef DECIMAL_T (*LOSS_T)(DECIMAL_T _out, DECIMAL_T _expected, bool _derived);
  typedef DECIMAL_T (*MUTATE_CALLBACK_T)();
}
