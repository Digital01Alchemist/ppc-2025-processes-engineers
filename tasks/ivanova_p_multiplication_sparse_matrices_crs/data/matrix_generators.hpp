#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

// Единичная матрица
inline CRSMatrix CreateIdentityMatrix(int n) {
  CRSMatrix identity_matrix;
  identity_matrix.n = n;
  identity_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  identity_matrix.row_ptr[0] = 0;
  for (int idx = 0; idx < n; ++idx) {
    identity_matrix.col_indices.push_back(idx);
    identity_matrix.values.push_back(1.0);
    identity_matrix.row_ptr[idx + 1] = idx + 1;
  }
  return identity_matrix;
}

// Нулевая матрица (полностью разреженная)
inline CRSMatrix CreateZeroMatrix(int n) {
  CRSMatrix zero_matrix;
  zero_matrix.n = n;
  zero_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
  return zero_matrix;
}

// Диагональная матрица с заданным значением
inline CRSMatrix CreateDiagonalMatrix(int n, double val) {
  CRSMatrix diagonal_matrix;
  diagonal_matrix.n = n;
  diagonal_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  diagonal_matrix.row_ptr[0] = 0;
  for (int idx = 0; idx < n; ++idx) {
    diagonal_matrix.col_indices.push_back(idx);
    diagonal_matrix.values.push_back(val);
    diagonal_matrix.row_ptr[idx + 1] = idx + 1;
  }
  return diagonal_matrix;
}

// Трёхдиагональная матрица
inline CRSMatrix CreateTridiagonalMatrix(int n, double sub, double diag, double sup) {
  CRSMatrix tridiagonal_matrix;
  tridiagonal_matrix.n = n;
  tridiagonal_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  int non_zero_count = 0;
  tridiagonal_matrix.row_ptr[0] = 0;

  for (int idx = 0; idx < n; ++idx) {
    if (idx > 0) {
      tridiagonal_matrix.col_indices.push_back(idx - 1);
      tridiagonal_matrix.values.push_back(sub);
      ++non_zero_count;
    }

    tridiagonal_matrix.col_indices.push_back(idx);
    tridiagonal_matrix.values.push_back(diag);
    ++non_zero_count;

    if (idx < n - 1) {
      tridiagonal_matrix.col_indices.push_back(idx + 1);
      tridiagonal_matrix.values.push_back(sup);
      ++non_zero_count;
    }

    tridiagonal_matrix.row_ptr[idx + 1] = non_zero_count;
  }
  return tridiagonal_matrix;
}

// Матрица с одним элементом
inline CRSMatrix CreateSingleElementMatrix(int n, int row, int col, double val) {
  CRSMatrix single_element_matrix;
  single_element_matrix.n = n;
  single_element_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);

  for (int idx = 0; idx <= n; ++idx) {
    single_element_matrix.row_ptr[idx] = (idx <= row) ? 0 : 1;
  }

  single_element_matrix.col_indices.push_back(col);
  single_element_matrix.values.push_back(val);
  return single_element_matrix;
}

// Верхнетреугольная матрица
inline CRSMatrix CreateUpperTriangularMatrix(int n, double val) {
  CRSMatrix upper_triangular_matrix;
  upper_triangular_matrix.n = n;
  upper_triangular_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  int non_zero_count = 0;
  upper_triangular_matrix.row_ptr[0] = 0;

  for (int row_idx = 0; row_idx < n; ++row_idx) {
    for (int col_idx = row_idx; col_idx < n; ++col_idx) {
      upper_triangular_matrix.col_indices.push_back(col_idx);
      upper_triangular_matrix.values.push_back(val);
      ++non_zero_count;
    }
    upper_triangular_matrix.row_ptr[row_idx + 1] = non_zero_count;
  }
  return upper_triangular_matrix;
}

// Нижнетреугольная матрица
inline CRSMatrix CreateLowerTriangularMatrix(int n, double val) {
  CRSMatrix lower_triangular_matrix;
  lower_triangular_matrix.n = n;
  lower_triangular_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  int non_zero_count = 0;
  lower_triangular_matrix.row_ptr[0] = 0;

  for (int row_idx = 0; row_idx < n; ++row_idx) {
    for (int col_idx = 0; col_idx <= row_idx; ++col_idx) {
      lower_triangular_matrix.col_indices.push_back(col_idx);
      lower_triangular_matrix.values.push_back(val);
      ++non_zero_count;
    }
    lower_triangular_matrix.row_ptr[row_idx + 1] = non_zero_count;
  }
  return lower_triangular_matrix;
}

// Случайная разреженная матрица
inline CRSMatrix CreateRandomSparseMatrix(int n, double density, unsigned int seed) {
  CRSMatrix random_matrix;
  random_matrix.n = n;
  random_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);

  std::mt19937 generator(seed);
  std::uniform_real_distribution<> probability_distribution(0.0, 1.0);
  std::uniform_real_distribution<> value_distribution(-10.0, 10.0);

  int non_zero_count = 0;
  random_matrix.row_ptr[0] = 0;

  for (int row_idx = 0; row_idx < n; ++row_idx) {
    for (int col_idx = 0; col_idx < n; ++col_idx) {
      if (probability_distribution(generator) < density) {
        random_matrix.col_indices.push_back(col_idx);
        random_matrix.values.push_back(value_distribution(generator));
        ++non_zero_count;
      }
    }
    random_matrix.row_ptr[row_idx + 1] = non_zero_count;
  }
  return random_matrix;
}

// Матрица с пустыми строками (только чётные строки имеют элементы)
inline CRSMatrix CreateMatrixWithEmptyRows(int n) {
  CRSMatrix matrix_with_empty_rows;
  matrix_with_empty_rows.n = n;
  matrix_with_empty_rows.row_ptr.resize(static_cast<std::size_t>(n) + 1);

  int non_zero_count = 0;
  matrix_with_empty_rows.row_ptr[0] = 0;

  for (int row_idx = 0; row_idx < n; ++row_idx) {
    if (row_idx % 2 == 0) {
      matrix_with_empty_rows.col_indices.push_back(row_idx);
      matrix_with_empty_rows.values.push_back(static_cast<double>(row_idx + 1));
      ++non_zero_count;
    }
    matrix_with_empty_rows.row_ptr[row_idx + 1] = non_zero_count;
  }
  return matrix_with_empty_rows;
}

// Антидиагональная матрица
inline CRSMatrix CreateAntiDiagonalMatrix(int n, double val) {
  CRSMatrix anti_diagonal_matrix;
  anti_diagonal_matrix.n = n;
  anti_diagonal_matrix.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  anti_diagonal_matrix.row_ptr[0] = 0;

  for (int idx = 0; idx < n; ++idx) {
    anti_diagonal_matrix.col_indices.push_back(n - 1 - idx);
    anti_diagonal_matrix.values.push_back(val);
    anti_diagonal_matrix.row_ptr[idx + 1] = idx + 1;
  }
  return anti_diagonal_matrix;
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
