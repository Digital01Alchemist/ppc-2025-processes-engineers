#include "ivanova_p_multiplication_sparse_matrices_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

// Вспомогательные функции
namespace {
void BroadcastMatrices(int n, int a_nnz, int b_nnz, CRSMatrix &local_a, CRSMatrix &local_b,
                       const std::tuple<CRSMatrix, CRSMatrix> &input) {
  local_a.n = n;
  local_a.values.resize(static_cast<std::size_t>(a_nnz));
  local_a.col_indices.resize(static_cast<std::size_t>(a_nnz));
  local_a.row_ptr.resize(static_cast<std::size_t>(n) + 1);

  local_b.n = n;
  local_b.values.resize(static_cast<std::size_t>(b_nnz));
  local_b.col_indices.resize(static_cast<std::size_t>(b_nnz));
  local_b.row_ptr.resize(static_cast<std::size_t>(n) + 1);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &[matrix_a, matrix_b] = input;
    local_a = matrix_a;
    local_b = matrix_b;
  }

  // Рассылаем данные матрицы A
  if (a_nnz > 0) {
    MPI_Bcast(local_a.values.data(), a_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_a.col_indices.data(), a_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(local_a.row_ptr.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем данные матрицы B
  if (b_nnz > 0) {
    MPI_Bcast(local_b.values.data(), b_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_b.col_indices.data(), b_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(local_b.row_ptr.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void ComputeLocalRows(int start_row, int count, const CRSMatrix &local_a, const CRSMatrix &local_b,
                      CRSMatrix &local_c) {
  local_c.n = local_a.n;
  if (count > 0) {
    local_c.row_ptr.resize(static_cast<std::size_t>(count) + 1);
    local_c.row_ptr[0] = 0;
  }

  std::unordered_map<int, double> accumulator;

  for (int i = 0; i < count; ++i) {
    const int global_row = start_row + i;
    accumulator.clear();

    for (int ia = local_a.row_ptr[global_row]; ia < local_a.row_ptr[global_row + 1]; ++ia) {
      const int k = local_a.col_indices[ia];
      const double a_value = local_a.values[ia];

      for (int ib = local_b.row_ptr[k]; ib < local_b.row_ptr[k + 1]; ++ib) {
        const int col = local_b.col_indices[ib];
        accumulator[col] += a_value * local_b.values[ib];
      }
    }

    // Собираем только ненулевые значения
    for (const auto &[col, val] : accumulator) {
      if (val != 0.0) {
        local_c.col_indices.push_back(col);
        local_c.values.push_back(val);
      }
    }

    if (count > 0) {
      local_c.row_ptr[i + 1] = static_cast<int>(local_c.values.size());
    }
  }
}

void GatherResults(int n, int rows_per_proc, int extra, int my_start, int count, int size, const CRSMatrix &local_c,
                   CRSMatrix &result) {
  result.n = n;
  result.row_ptr.resize(static_cast<std::size_t>(n) + 1);
  result.row_ptr[0] = 0;

  // Копируем свою часть (процесс 0)
  result.values = local_c.values;
  result.col_indices = local_c.col_indices;

  // Заполняем row_ptr для своей части
  for (int i = 0; i < count; ++i) {
    result.row_ptr[my_start + i + 1] = local_c.row_ptr[i + 1];
  }

  // Текущее смещение для следующих процессов
  int current_base = static_cast<int>(result.values.size());

  // Получаем данные от других процессов
  for (int proc = 1; proc < size; ++proc) {
    const int proc_start = proc * rows_per_proc + std::min(proc, extra);
    const int proc_count = rows_per_proc + (proc < extra ? 1 : 0);

    if (proc_count == 0) {
      continue;
    }

    // Получаем количество ненулевых элементов
    int proc_nnz = 0;
    MPI_Recv(&proc_nnz, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Получаем row_ptr
    std::vector<int> proc_row_ptr(static_cast<std::size_t>(proc_count) + 1);
    MPI_Recv(proc_row_ptr.data(), proc_count + 1, MPI_INT, proc, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (proc_nnz > 0) {
      // Получаем значения и индексы
      std::vector<double> proc_values(static_cast<std::size_t>(proc_nnz));
      std::vector<int> proc_cols(static_cast<std::size_t>(proc_nnz));

      MPI_Recv(proc_values.data(), proc_nnz, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(proc_cols.data(), proc_nnz, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Конвертируем локальные индексы в глобальные
      for (int proc_i = 0; proc_i < proc_count; ++proc_i) {
        result.row_ptr[proc_start + proc_i + 1] = current_base + proc_row_ptr[proc_i + 1];
      }

      result.values.insert(result.values.end(), proc_values.begin(), proc_values.end());
      result.col_indices.insert(result.col_indices.end(), proc_cols.begin(), proc_cols.end());
      current_base += proc_nnz;
    } else {
      // Все строки пустые
      for (int proc_i = 0; proc_i < proc_count; ++proc_i) {
        result.row_ptr[proc_start + proc_i + 1] = current_base;
      }
    }
  }
}

void SendResultsToRoot(int count, const CRSMatrix &local_c) {
  if (count > 0) {
    const int local_nnz = static_cast<int>(local_c.values.size());

    // Отправляем количество ненулевых элементов
    MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    // Отправляем row_ptr
    if (!local_c.row_ptr.empty()) {
      MPI_Send(local_c.row_ptr.data(), count + 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    // Отправляем значения и индексы только если они есть
    if (local_nnz > 0) {
      MPI_Send(local_c.values.data(), local_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Send(local_c.col_indices.data(), local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
  }
}
}  // namespace

IvanovaPMultiplicationSparseMatricesCrsMPI::IvanovaPMultiplicationSparseMatricesCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int valid = 0;
  if (rank == 0) {
    const auto &[matrix_a, matrix_b] = GetInput();
    valid = (matrix_a.IsValid() && matrix_b.IsValid() && matrix_a.n == matrix_b.n) ? 1 : 0;
  }

  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid != 0;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = CRSMatrix{};
  }
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Рассылаем размеры матриц
  int n = 0;
  int a_nnz = 0;
  int b_nnz = 0;

  if (rank == 0) {
    const auto &[matrix_a, matrix_b] = GetInput();
    n = matrix_a.n;
    a_nnz = static_cast<int>(matrix_a.values.size());
    b_nnz = static_cast<int>(matrix_b.values.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Локальные копии матриц
  CRSMatrix local_a;
  CRSMatrix local_b;

  // Передаем входные данные во вспомогательную функцию
  BroadcastMatrices(n, a_nnz, b_nnz, local_a, local_b, GetInput());

  // Распределяем строки матрицы A
  const int rows_per_proc = n / size;
  const int extra = n % size;
  const int my_start = (rank * rows_per_proc) + std::min(rank, extra);
  const int count = rows_per_proc + (rank < extra ? 1 : 0);

  // Локальная часть матрицы C
  CRSMatrix local_c;
  ComputeLocalRows(my_start, count, local_a, local_b, local_c);

  // ---------- Собираем результаты на процессе 0 ----------
  if (rank == 0) {
    GatherResults(n, rows_per_proc, extra, my_start, count, size, local_c, GetOutput());
  } else {
    SendResultsToRoot(count, local_c);
  }

  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank != 0 || GetOutput().IsValid();
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
