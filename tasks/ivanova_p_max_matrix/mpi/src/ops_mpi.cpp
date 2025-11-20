#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>   // для size_t
#include <iostream>  // для std::cout, если используется
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixMPI::IvanovaPMaxMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  // Используем reserve и push_back для безопасного копирования
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput() = std::numeric_limits<int>::min();
}

bool IvanovaPMaxMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;  // Пустая матрица невалидна
  }

  size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {  // range-based for
    if (row.empty() || row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool IvanovaPMaxMatrixMPI::PreProcessingImpl() {
  // Такая же инициализация
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool IvanovaPMaxMatrixMPI::RunImpl() {
  const auto &input = GetInput();

  int world_size = 0;  // Исправлено: раздельная инициализация
  int world_rank = 0;  // Исправлено: раздельная инициализация
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows = 0;
  int cols = 0;  // Исправлено: раздельная инициализация

  if (world_rank == 0) {
    rows = static_cast<int>(input.size());
    cols = (rows > 0 ? static_cast<int>(input[0].size()) : 0);  // Исправлено: явное приведение
  }

  // Рассылаем размеры всем
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }

  // -------------------------------------------
  // Готовим распределение по строкам
  // -------------------------------------------
  std::vector<int> rows_per_rank(world_size, 0);
  int base = rows / world_size;
  int rem = rows % world_size;

  for (int rank_index = 0; rank_index < world_size; ++rank_index) {  // Исправлено: более длинное имя
    rows_per_rank[rank_index] = base + (rank_index < rem ? 1 : 0);
  }

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  for (int rank_index = 0; rank_index < world_size; ++rank_index) {  // Исправлено: более длинное имя
    sendcounts[rank_index] = rows_per_rank[rank_index] * cols;
    displs[rank_index] = (rank_index == 0 ? 0 : displs[rank_index - 1] + sendcounts[rank_index - 1]);
  }

  // -------------------------------------------
  // Root упаковывает матрицу в flat-буфер
  // -------------------------------------------
  std::vector<int> flat;
  if (world_rank == 0) {
    flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));  // Исправлено: явное приведение
    int pos = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        flat[pos++] = input[i][j];
      }
    }
  }

  // -------------------------------------------
  // Каждый процесс получает свой кусок
  // -------------------------------------------
  int my_count = sendcounts[world_rank];
  std::vector<int> local_flat(my_count);

  MPI_Scatterv(world_rank == 0 ? flat.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_flat.data(),
               my_count, MPI_INT, 0, MPI_COMM_WORLD);

  // -------------------------------------------
  // Находим локальный максимум
  // -------------------------------------------
  int local_max = std::numeric_limits<int>::min();
  for (int value : local_flat) {
    local_max = std::max(value, local_max);  // Исправлено: std::max вместо ручной проверки
  }

  // -------------------------------------------
  // Собираем глобальный максимум
  // -------------------------------------------
  int global_max = std::numeric_limits<int>::min();
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  // Раздаём обратно (по желанию)
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool IvanovaPMaxMatrixMPI::PostProcessingImpl() {
  return true;
}
}  // namespace ivanova_p_max_matrix
