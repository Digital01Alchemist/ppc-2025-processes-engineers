#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>  // Добавлено для size_t
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"
// Убрал "util/include/util.hpp" так как он не используется

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixSEQ::IvanovaPMaxMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  // Безопасная инициализация
  GetInput().clear();
  if (!in.empty()) {
    GetInput() = in;
  }

  GetOutput() = std::numeric_limits<int>::min();
}

bool IvanovaPMaxMatrixSEQ::ValidationImpl() {
  // Базовая проверка на пустоту
  if (GetInput().empty()) {
    return false;
  }

  // Проверяем все строки
  size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {  // Исправлено: range-based for
    if (row.size() != cols) {
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
  for (const auto &row : GetInput()) {  // Исправлено: range-based for для внешнего цикла
    for (int val : row) {
      max_val = std::max(val, max_val);  // Исправлено: std::max вместо ручной проверки
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
