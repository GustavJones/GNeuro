#pragma once
#include <string>
#include "GMath/Matrix.hpp"
#include "GNeuro/Activation.hpp"
#include "GNeuro/Loss.hpp"

namespace GNeuro {
  template<typename value_t>
  struct FunctionType {
    typedef GMath::Matrix<value_t> (*activation_t)(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName);
    typedef GMath::Matrix<value_t> (*loss_t)(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName);

    inline static const GMath::DynamicArray<activation_t> &GetActivationFunctions() {
			static GMath::DynamicArray<activation_t> activation = {
				GNeuro::None,
				GNeuro::Sigmoid,
				GNeuro::ReLu,
				GNeuro::LeakyReLu,
				GNeuro::TanH,
				GNeuro::Softmax,
			};

      return activation;
		}

		inline static const GMath::DynamicArray<loss_t> &GetLossFunctions() {
			static GMath::DynamicArray<loss_t> loss = {
				GNeuro::Error,
				GNeuro::NegativeError,
				GNeuro::SquaredError,
				GNeuro::SquaredNegativeError,
			};

			return loss;
		}
  };
}
