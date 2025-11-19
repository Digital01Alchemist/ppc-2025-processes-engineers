#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixSEQ::IvanovaPMaxMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());           // Нужно в любом случае!!!
  GetInput() = in;                                // и тут, но не знач
  GetOutput() = std::numeric_limits<int>::min();  // и тут, но не знач
}

bool IvanovaPMaxMatrixSEQ::ValidationImpl() {
  // Базовая проверка на пустоту
  if (GetInput().empty()) {
    return false;
  }

  // Проверяем все строки
  int cols = GetInput()[0].size();
  for (size_t i = 0; i < GetInput().size(); i++) {
    if (GetInput()[i].size() != cols) {
      return false;
    }
  }

  return true;
}

bool IvanovaPMaxMatrixSEQ::PreProcessingImpl() {
  // Инициализируем выход минимальным значением
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool IvanovaPMaxMatrixSEQ::RunImpl() {
  int max_val = std::numeric_limits<int>::min();

  // Простой двойной цикл по матрице
  for (size_t i = 0; i < GetInput().size(); i++) {
    for (size_t j = 0; j < GetInput()[i].size(); j++) {
      if (GetInput()[i][j] > max_val) {
        max_val = GetInput()[i][j];
      }
    }
  }

  GetOutput() = max_val;
  return true;
}

bool IvanovaPMaxMatrixSEQ::PostProcessingImpl() {
  // Дополнительная обработка не требуется
  return true;
}

}  // namespace ivanova_p_max_matrix
