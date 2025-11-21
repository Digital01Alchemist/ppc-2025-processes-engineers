#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixMPI::IvanovaPMaxMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput() = std::numeric_limits<int>::min();
}

bool IvanovaPMaxMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  const size_t cols = GetInput()[0].size();
  return std::all_of(GetInput().begin(), GetInput().end(),
                     [cols](const std::vector<int> &row) { return !row.empty() && row.size() == cols; });
}

bool IvanovaPMaxMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

// Вспомогательные приватные методы для уменьшения сложности RunImpl
namespace {

std::tuple<std::vector<int>, std::vector<int>> CalculateDistribution(int rows, int cols, int world_size) {
  std::vector<int> rows_per_rank(world_size, 0);
  const int base = rows / world_size;
  const int rem = rows % world_size;

  for (int rank_index = 0; rank_index < world_size; ++rank_index) {
    rows_per_rank[rank_index] = base + (rank_index < rem ? 1 : 0);
  }

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
    sendcounts[rank_idx] = rows_per_rank[rank_idx] * cols;
    displs[rank_idx] = (rank_idx == 0 ? 0 : displs[rank_idx - 1] + sendcounts[rank_idx - 1]);
  }

  return {sendcounts, displs};
}

std::vector<int> PackMatrix(const InType &input, int rows, int cols) {
  std::vector<int> flat(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  int pos = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      flat[pos++] = input[i][j];
    }
  }
  return flat;
}

int FindLocalMax(const std::vector<int> &local_flat) {
  int local_max = std::numeric_limits<int>::min();
  for (int value : local_flat) {
    local_max = std::max(value, local_max);
  }
  return local_max;
}

}  // namespace

bool IvanovaPMaxMatrixMPI::RunImpl() {
  const auto &input = GetInput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Раздельные объявления
  int rows = 0;
  int cols = 0;

  if (world_rank == 0) {
    rows = static_cast<int>(input.size());
    cols = (rows > 0 ? static_cast<int>(input[0].size()) : 0);  // Исправлено: явное приведение типа
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }

  // Вычисление распределения данных
  auto [sendcounts, displs] = CalculateDistribution(rows, cols, world_size);

  // Упаковка матрицы (только на процессе 0)
  std::vector<int> flat;
  if (world_rank == 0) {
    flat = PackMatrix(input, rows, cols);
  }

  // Раздача данных по процессам
  const int my_count = sendcounts[world_rank];
  std::vector<int> local_flat(my_count);

  MPI_Scatterv(world_rank == 0 ? flat.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_flat.data(),
               my_count, MPI_INT, 0, MPI_COMM_WORLD);

  // Поиск локального максимума
  const int local_max = FindLocalMax(local_flat);

  // Сбор глобального максимума
  int global_max = std::numeric_limits<int>::min();
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool IvanovaPMaxMatrixMPI::PostProcessingImpl() {
  return true;
}
}  // namespace ivanova_p_max_matrix
