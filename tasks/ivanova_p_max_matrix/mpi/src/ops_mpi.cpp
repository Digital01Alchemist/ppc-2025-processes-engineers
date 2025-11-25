#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>  // memcpy
#include <limits>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix {

IvanovaPMaxMatrixMPI::IvanovaPMaxMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::numeric_limits<int>::min();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;  // Только root хранит входную матрицу
  } else {
    GetInput().clear();  // Остальные — пустая матрица
  }
}

bool IvanovaPMaxMatrixMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool ok = true;

  if (rank == 0) {
    if (GetInput().empty()) {
      ok = false;
    } else {
      const size_t cols = GetInput()[0].size();
      for (const auto &row : GetInput()) {
        if (row.size() != cols) {
          ok = false;
          break;
        }
      }
    }
  }

  int ok_int = ok ? 1 : 0;
  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int != 0;
}

bool IvanovaPMaxMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}
namespace {

// Универсальная функция поиска максимума по указателю и размеру
// (позволяет не создавать лишние вектора)
int FindMaxInPointer(const int *data, size_t size) {
  if (size == 0) {
    return std::numeric_limits<int>::min();
  }
  // Используем стандартный алгоритм для raw-памяти
  // Можно использовать std::max_element
  int max_val = data[0];
  for (size_t i = 1; i < size; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
    }
  }
  return max_val;
}

// Твоя функция PackMatrix осталась без изменений, она хорошая
std::vector<int> PackMatrix(const std::vector<std::vector<int>> &input, int rows, int cols) {
  if (rows == 0 || cols == 0) {
    return {};
  }
  std::vector<int> flat(static_cast<size_t>(rows) * cols);
  size_t pos = 0;
  for (int i = 0; i < rows; ++i) {
    std::memcpy(&flat[pos], input[i].data(), sizeof(int) * cols);
    pos += cols;
  }
  return flat;
}

}  // namespace

bool IvanovaPMaxMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // --- ОПТИМИЗАЦИЯ 1: Если процесс один, не тратим время на MPI и копирование ---
  if (size == 1) {
    if (GetInput().empty()) {
      GetOutput() = std::numeric_limits<int>::min();
      return true;
    }
    // Честный последовательный поиск без аллокаций плоского массива
    int global_max = std::numeric_limits<int>::min();
    bool first = true;
    for (const auto &row : GetInput()) {
      if (row.empty()) {
        continue;
      }
      int row_max = FindMaxInPointer(row.data(), row.size());
      if (first || row_max > global_max) {
        global_max = row_max;
        first = false;
      }
    }
    GetOutput() = global_max;
    return true;
  }
  // -----------------------------------------------------------------------------

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = rows > 0 ? static_cast<int>(GetInput()[0].size()) : 0;
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Если матрица пустая, ранний выход
  if (rows == 0 || cols == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }

  const int total = rows * cols;
  int base = total / size;
  int rem = total % size;

  // Рассчитываем, сколько элементов обрабатывает текущий процесс
  int my_count = base + (rank < rem ? 1 : 0);

  // Подготовка данных для scatter (нужна всем для displs/counts,
  // хотя worker-ам нужны только counts для Scatterv, но для логики оставим)
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    displs[0] = 0;
    for (int i = 0; i < size; i++) {
      sendcounts[i] = base + (i < rem ? 1 : 0);
      if (i > 0) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
      }
    }
  }

  std::vector<int> flat;
  std::vector<int> local;  // Вектор для worker-ов
  int local_max = std::numeric_limits<int>::min();

  if (rank == 0) {
    flat = PackMatrix(GetInput(), rows, cols);

    // --- ОПТИМИЗАЦИЯ 2: MPI_IN_PLACE на Root ---
    // Root не выделяет память под local и не копирует в него данные.
    // Он использует MPI_IN_PLACE, чтобы сказать Scatterv: "Мои данные уже у меня".
    MPI_Scatterv(flat.data(), sendcounts.data(), displs.data(), MPI_INT, MPI_IN_PLACE, my_count, MPI_INT, 0,
                 MPI_COMM_WORLD);

    // Root считает максимум прямо внутри flat массива
    // Его данные лежат с displs[0] (это 0) и имеют длину sendcounts[0] (это my_count)
    local_max = FindMaxInPointer(flat.data(), my_count);

  } else {
    // Worker-ы выделяют память
    local.resize(my_count);

    // Worker-ы получают данные как обычно
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,  // send-аргументы игнорируются на worker
                 local.data(), my_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Worker считает по своему локальному вектору
    local_max = FindMaxInPointer(local.data(), my_count);
  }

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool IvanovaPMaxMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ivanova_p_max_matrix
