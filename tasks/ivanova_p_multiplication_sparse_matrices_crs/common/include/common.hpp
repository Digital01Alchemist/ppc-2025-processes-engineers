#pragma once

#include <cstddef>  // для size_t
#include <string>   // для std::string
#include <tuple>    // для std::tuple
#include <unordered_map>
#include <vector>

#include "task/include/task.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

// -------------------- CRS MATRIX --------------------

struct CRSMatrix {
  int n = 0;
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;

  [[nodiscard]] bool IsValid() const {
    if (n <= 0) {
      return false;
    }
    if (row_ptr.size() != static_cast<std::size_t>(n) + 1) {
      return false;
    }
    if (values.size() != col_indices.size()) {
      return false;
    }
    // Используем явное приведение типов для сравнения
    const auto last_row_ptr = row_ptr.back();
    const auto values_size = static_cast<int>(values.size());
    if (last_row_ptr != values_size) {
      return false;
    }
    return true;
  }

  bool operator==(const CRSMatrix &other) const {
    return n == other.n && values == other.values && col_indices == other.col_indices && row_ptr == other.row_ptr;
  }
};

// -------------------- TYPES --------------------

using InType = std::tuple<CRSMatrix, CRSMatrix>;  // матрицы a и b
using OutType = CRSMatrix;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

// -------------------- CRS × CRS --------------------

inline CRSMatrix MultiplyCRS(const CRSMatrix &matrix_a, const CRSMatrix &matrix_b) {
  CRSMatrix result;
  result.n = matrix_a.n;
  result.row_ptr.resize(static_cast<std::size_t>(result.n) + 1);
  result.row_ptr[0] = 0;

  std::unordered_map<int, double> accumulator;

  for (int i = 0; i < matrix_a.n; ++i) {
    accumulator.clear();

    for (int ia = matrix_a.row_ptr[i]; ia < matrix_a.row_ptr[i + 1]; ++ia) {
      const int k = matrix_a.col_indices[ia];
      const double a_value = matrix_a.values[ia];

      for (int ib = matrix_b.row_ptr[k]; ib < matrix_b.row_ptr[k + 1]; ++ib) {
        const int col = matrix_b.col_indices[ib];
        accumulator[col] += a_value * matrix_b.values[ib];
      }
    }

    // Собираем только ненулевые значения
    for (const auto &[col, val] : accumulator) {
      if (val != 0.0) {
        result.col_indices.push_back(col);
        result.values.push_back(val);
      }
    }

    result.row_ptr[i + 1] = static_cast<int>(result.values.size());
  }

  return result;
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
