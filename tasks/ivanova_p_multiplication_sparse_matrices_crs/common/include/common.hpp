#pragma once

#include <tuple>
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

  bool IsValid() const {
    return n > 0 && row_ptr.size() == static_cast<size_t>(n + 1) && values.size() == col_indices.size() &&
           row_ptr.back() == static_cast<int>(values.size());
  }

  bool operator==(const CRSMatrix &other) const {
    return n == other.n && values == other.values && col_indices == other.col_indices && row_ptr == other.row_ptr;
  }
};

// -------------------- TYPES --------------------

using InType = std::tuple<CRSMatrix, CRSMatrix>;  // A, B
using OutType = CRSMatrix;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

// -------------------- CRS × CRS --------------------

inline CRSMatrix MultiplyCRS(const CRSMatrix &A, const CRSMatrix &B) {
  CRSMatrix C;
  C.n = A.n;
  C.row_ptr.resize(C.n + 1);
  C.row_ptr[0] = 0;

  std::unordered_map<int, double> acc;
  std::vector<int> used_cols;

  for (int i = 0; i < A.n; i++) {
    acc.clear();
    used_cols.clear();

    for (int ia = A.row_ptr[i]; ia < A.row_ptr[i + 1]; ia++) {
      int k = A.col_indices[ia];
      double a = A.values[ia];

      for (int ib = B.row_ptr[k]; ib < B.row_ptr[k + 1]; ib++) {
        int col = B.col_indices[ib];
        auto &val = acc[col];
        if (val == 0.0) {
          used_cols.push_back(col);
        }
        val += a * B.values[ib];
      }
    }

    for (int col : used_cols) {
      double val = acc[col];
      if (val != 0.0) {
        C.col_indices.push_back(col);
        C.values.push_back(val);
      }
    }

    C.row_ptr[i + 1] = static_cast<int>(C.values.size());
  }

  return C;
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
