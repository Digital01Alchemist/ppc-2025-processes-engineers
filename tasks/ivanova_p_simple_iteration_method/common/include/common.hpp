#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace ivanova_p_simple_iteration_method {

using InType = int;   // Размер системы
using OutType = int;  // Сумма компонент решения (округляется)
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ivanova_p_simple_iteration_method
