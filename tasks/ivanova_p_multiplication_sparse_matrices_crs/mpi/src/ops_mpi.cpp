#include "ivanova_p_multiplication_sparse_matrices_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <unordered_map>
#include <vector>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

IvanovaPMultiplicationSparseMatricesCrsMPI::IvanovaPMultiplicationSparseMatricesCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int valid = 0;
  if (rank == 0) {
    const auto &[A, B] = GetInput();
    valid = (A.IsValid() && B.IsValid() && A.n == B.n) ? 1 : 0;
  }

  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::PreProcessingImpl() {
  if (int rank = 0; (MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank == 0)) {
    GetOutput() = CRSMatrix{};
  }
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Рассылаем размеры матриц
  int n = 0;
  int a_nnz = 0;
  int b_nnz = 0;

  if (rank == 0) {
    const auto &[A, B] = GetInput();
    n = A.n;
    a_nnz = static_cast<int>(A.values.size());
    b_nnz = static_cast<int>(B.values.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Локальные копии матриц A и B
  CRSMatrix localA;
  CRSMatrix localB;

  localA.n = n;
  localA.values.resize(a_nnz);
  localA.col_indices.resize(a_nnz);
  localA.row_ptr.resize(n + 1);

  localB.n = n;
  localB.values.resize(b_nnz);
  localB.col_indices.resize(b_nnz);
  localB.row_ptr.resize(n + 1);

  if (rank == 0) {
    const auto &[A, B] = GetInput();
    localA = A;
    localB = B;
  }

  // Рассылаем данные A
  if (a_nnz > 0) {
    MPI_Bcast(localA.values.data(), a_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(localA.col_indices.data(), a_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(localA.row_ptr.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем данные B
  if (b_nnz > 0) {
    MPI_Bcast(localB.values.data(), b_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(localB.col_indices.data(), b_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(localB.row_ptr.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределяем строки матрицы A
  int rows_per_proc = n / size;
  int extra = n % size;
  int my_start = rank * rows_per_proc + std::min(rank, extra);
  int count = rows_per_proc + (rank < extra ? 1 : 0);

  // Локальная часть матрицы C
  CRSMatrix localC;
  localC.n = n;
  if (count > 0) {
    localC.row_ptr.resize(static_cast<size_t>(count) + 1);
    localC.row_ptr[0] = 0;
  }

  // Умножаем локальные строки (исправленная версия)
  std::unordered_map<int, double> acc;

  for (int i = 0; i < count; i++) {
    int global_row = my_start + i;
    acc.clear();

    for (int ia = localA.row_ptr[global_row]; ia < localA.row_ptr[global_row + 1]; ia++) {
      int k = localA.col_indices[ia];
      double a = localA.values[ia];

      for (int ib = localB.row_ptr[k]; ib < localB.row_ptr[k + 1]; ib++) {
        int col = localB.col_indices[ib];
        acc[col] += a * localB.values[ib];
      }
    }

    // Собираем только ненулевые значения (исправление проблемы с used_cols)
    for (const auto &[col, val] : acc) {
      if (val != 0.0) {  // Можно использовать std::abs(val) > 1e-12 для учёта погрешностей
        localC.col_indices.push_back(col);
        localC.values.push_back(val);
      }
    }

    if (count > 0) {
      localC.row_ptr[i + 1] = static_cast<int>(localC.values.size());
    }
  }

  // ---------- Собираем результаты на процессе 0 ----------
  if (rank == 0) {
    CRSMatrix &C = GetOutput();
    C.n = n;
    C.row_ptr.resize(static_cast<size_t>(n) + 1);
    C.row_ptr[0] = 0;

    // Копируем свою часть
    std::copy(localC.values.begin(), localC.values.end(), std::back_inserter(C.values));
    std::copy(localC.col_indices.begin(), localC.col_indices.end(), std::back_inserter(C.col_indices));

    // Заполняем row_ptr для своей части
    for (int i = 0; i < count; i++) {
      C.row_ptr[my_start + i + 1] = localC.row_ptr[i + 1];
    }

    // Получаем данные от других процессов
    for (int p = 1; p < size; p++) {
      int p_start = p * rows_per_proc + std::min(p, extra);
      int p_count = rows_per_proc + (p < extra ? 1 : 0);

      // Критическое исправление: пропускаем процессы, которым не досталось строк
      if (p_count == 0) {
        continue;
      }

      // Получаем количество ненулевых элементов
      int p_nnz = 0;
      MPI_Recv(&p_nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Получаем row_ptr
      std::vector<int> p_row_ptr(static_cast<size_t>(p_count) + 1);
      MPI_Recv(p_row_ptr.data(), p_count + 1, MPI_INT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (p_nnz > 0) {
        std::vector<double> p_vals(p_nnz);
        std::vector<int> p_cols(p_nnz);

        MPI_Recv(p_vals.data(), p_nnz, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p_cols.data(), p_nnz, MPI_INT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Конвертируем локальные индексы в глобальные
        int base = static_cast<int>(C.values.size());
        for (int p_i = 0; p_i < p_count; p_i++) {
          C.row_ptr[p_start + p_i + 1] = base + p_row_ptr[p_i + 1];
        }

        C.values.insert(C.values.end(), p_vals.begin(), p_vals.end());
        C.col_indices.insert(C.col_indices.end(), p_cols.begin(), p_cols.end());
      } else {
        // Нет ненулевых элементов
        for (int p_i = 0; p_i < p_count; p_i++) {
          C.row_ptr[p_start + p_i + 1] = static_cast<int>(C.values.size());
        }
      }
    }
  } else {
    // Отправляем данные на процесс 0
    // Критическое исправление: отправляем данные ТОЛЬКО если есть что отправлять
    if (count > 0) {
      int local_nnz = static_cast<int>(localC.values.size());
      MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(localC.row_ptr.data(), count + 1, MPI_INT, 0, 3, MPI_COMM_WORLD);

      if (local_nnz > 0) {
        MPI_Send(localC.values.data(), local_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(localC.col_indices.data(), local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
      }
    }
  }

  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank != 0 || GetOutput().IsValid();
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
