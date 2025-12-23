#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

// Единичная матрица
inline CRSMatrix CreateIdentityMatrix(int n) {
  CRSMatrix I;
  I.n = n;
  I.row_ptr.resize(n + 1);
  I.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    I.col_indices.push_back(i);
    I.values.push_back(1.0);
    I.row_ptr[i + 1] = i + 1;
  }
  return I;
}

// Нулевая матрица (полностью разреженная)
inline CRSMatrix CreateZeroMatrix(int n) {
  CRSMatrix Z;
  Z.n = n;
  Z.row_ptr.resize(n + 1, 0);
  return Z;
}

// Диагональная матрица с заданным значением
inline CRSMatrix CreateDiagonalMatrix(int n, double val) {
  CRSMatrix D;
  D.n = n;
  D.row_ptr.resize(n + 1);
  D.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    D.col_indices.push_back(i);
    D.values.push_back(val);
    D.row_ptr[i + 1] = i + 1;
  }
  return D;
}

// Трёхдиагональная матрица
inline CRSMatrix CreateTridiagonalMatrix(int n, double sub, double diag, double sup) {
  CRSMatrix T;
  T.n = n;
  T.row_ptr.resize(n + 1);
  int nnz = 0;
  T.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    if (i > 0) {
      T.col_indices.push_back(i - 1);
      T.values.push_back(sub);
      nnz++;
    }
    T.col_indices.push_back(i);
    T.values.push_back(diag);
    nnz++;
    if (i < n - 1) {
      T.col_indices.push_back(i + 1);
      T.values.push_back(sup);
      nnz++;
    }
    T.row_ptr[i + 1] = nnz;
  }
  return T;
}

// Матрица с одним элементом
inline CRSMatrix CreateSingleElementMatrix(int n, int row, int col, double val) {
  CRSMatrix S;
  S.n = n;
  S.row_ptr.resize(n + 1, 0);
  for (int i = 0; i <= n; i++) {
    S.row_ptr[i] = (i <= row) ? 0 : 1;
  }
  S.col_indices.push_back(col);
  S.values.push_back(val);
  return S;
}

// Верхнетреугольная матрица
inline CRSMatrix CreateUpperTriangularMatrix(int n, double val) {
  CRSMatrix U;
  U.n = n;
  U.row_ptr.resize(n + 1);
  int nnz = 0;
  U.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      U.col_indices.push_back(j);
      U.values.push_back(val);
      nnz++;
    }
    U.row_ptr[i + 1] = nnz;
  }
  return U;
}

// Нижнетреугольная матрица
inline CRSMatrix CreateLowerTriangularMatrix(int n, double val) {
  CRSMatrix L;
  L.n = n;
  L.row_ptr.resize(n + 1);
  int nnz = 0;
  L.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      L.col_indices.push_back(j);
      L.values.push_back(val);
      nnz++;
    }
    L.row_ptr[i + 1] = nnz;
  }
  return L;
}

// Случайная разреженная матрица
inline CRSMatrix CreateRandomSparseMatrix(int n, double density, unsigned int seed) {
  CRSMatrix R;
  R.n = n;
  R.row_ptr.resize(n + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> prob(0.0, 1.0);
  std::uniform_real_distribution<> val(-10.0, 10.0);

  int nnz = 0;
  R.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (prob(gen) < density) {
        R.col_indices.push_back(j);
        R.values.push_back(val(gen));
        nnz++;
      }
    }
    R.row_ptr[i + 1] = nnz;
  }
  return R;
}

// Матрица с пустыми строками (только чётные строки имеют элементы)
inline CRSMatrix CreateMatrixWithEmptyRows(int n) {
  CRSMatrix M;
  M.n = n;
  M.row_ptr.resize(n + 1);
  int nnz = 0;
  M.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    if (i % 2 == 0) {
      M.col_indices.push_back(i);
      M.values.push_back(static_cast<double>(i + 1));
      nnz++;
    }
    M.row_ptr[i + 1] = nnz;
  }
  return M;
}

// Антидиагональная матрица
inline CRSMatrix CreateAntiDiagonalMatrix(int n, double val) {
  CRSMatrix A;
  A.n = n;
  A.row_ptr.resize(n + 1);
  A.row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    A.col_indices.push_back(n - 1 - i);
    A.values.push_back(val);
    A.row_ptr[i + 1] = i + 1;
  }
  return A;
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
