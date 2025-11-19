#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixMPI::IvanovaPMaxMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::numeric_limits<int>::min();
}

bool IvanovaPMaxMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;  // Пустая матрица невалидна
  }

  int cols = GetInput()[0].size();
  for (size_t i = 0; i < GetInput().size(); i++) {
    if (GetInput()[i].empty() || GetInput()[i].size() != static_cast<size_t>(cols)) {
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

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows = 0, cols = 0;

  if (world_rank == 0) {
    rows = input.size();
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

  for (int r = 0; r < world_size; ++r) {
    rows_per_rank[r] = base + (r < rem ? 1 : 0);
  }

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  for (int r = 0; r < world_size; ++r) {
    sendcounts[r] = rows_per_rank[r] * cols;
    displs[r] = (r == 0 ? 0 : displs[r - 1] + sendcounts[r - 1]);
  }

  // -------------------------------------------
  // Root упаковывает матрицу в flat-буфер
  // -------------------------------------------
  std::vector<int> flat;
  if (world_rank == 0) {
    flat.resize(rows * cols);
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
  for (int v : local_flat) {
    if (v > local_max) {
      local_max = v;
    }
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
