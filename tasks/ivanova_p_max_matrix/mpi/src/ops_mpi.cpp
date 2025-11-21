#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>  // Добавлено для size_t
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

  const size_t cols = GetInput()[0].size();
  // Используем std::all_of для проверки всех строк
  return std::all_of(GetInput().begin(), GetInput().end(),
                     [cols](const std::vector<int> &row) { return !row.empty() && row.size() == cols; });
}

bool IvanovaPMaxMatrixMPI::PreProcessingImpl() {
  // Такая же инициализация
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool IvanovaPMaxMatrixMPI::RunImpl() {
  const auto &input = GetInput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows = 0;
  int cols = 0;

  if (world_rank == 0) {
    rows = static_cast<int>(input.size());
    cols = (rows > 0 ? input[0].size() : 0);
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

  for (int rank = 0; rank < world_size; ++rank) {
    rows_per_rank[rank] = base + (rank < rem ? 1 : 0);
  }

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  for (int rank = 0; rank < world_size; ++rank) {
    sendcounts[rank] = rows_per_rank[rank] * cols;
    displs[rank] = (rank == 0 ? 0 : displs[rank - 1] + sendcounts[rank - 1]);
  }

  // -------------------------------------------
  // Root упаковывает матрицу в flat-буфер
  // -------------------------------------------
  std::vector<int> flat;
  if (world_rank == 0) {
    flat.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
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
    local_max = std::max(value, local_max);
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
