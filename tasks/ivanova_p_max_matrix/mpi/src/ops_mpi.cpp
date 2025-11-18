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
  // Такая же проверка как в sequential версии
  if (GetInput().empty() || GetInput()[0].empty()) {
    return false;
  }

  int cols = GetInput()[0].size();
  for (int i = 0; i < GetInput().size(); i++) {
    if (GetInput()[i].size() != cols) {
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
  // Входная матрица (возможно, создана на каждом ранке тестовой инфраструктурой).
  const auto &input_matrix = GetInput();

  // Проверка MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  int world_size = 1;
  int world_rank = 0;
  if (mpi_initialized) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  }

  // Локальная копия матрицы, которой будем оперировать.
  std::vector<std::vector<int>> matrix_local;

  if (mpi_initialized) {
    // Rank 0 сериализует (flatten) матрицу и вещает её всем, чтобы
    // гарантировать единообразие данных между процессами.
    int rows = 0, cols = 0;
    if (world_rank == 0) {
      rows = static_cast<int>(input_matrix.size());
      cols = (rows > 0) ? static_cast<int>(input_matrix[0].size()) : 0;
    }

    // Сначала вещаем размеры
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rows < 0 || cols < 0) {
      return false;  // защита от неверных размеров
    }

    // Подготавливаем flat-буфер
    std::vector<int> flat;
    if (world_rank == 0) {
      flat.reserve(rows * cols);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          flat.push_back(input_matrix[i][j]);
        }
      }
    } else {
      flat.assign(rows * cols, std::numeric_limits<int>::min());
    }

    // Вещаем flat-буфер (если есть элементы)
    if (!flat.empty()) {
      MPI_Bcast(flat.data(), static_cast<int>(flat.size()), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Восстанавливаем локальную матрицу
    matrix_local.assign(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_local[i][j] = flat[i * cols + j];
      }
    }
  } else {
    // Без MPI — используем вход как локальную матрицу
    matrix_local = input_matrix;
  }

  const int rows_local = static_cast<int>(matrix_local.size());
  const int cols_local = (rows_local > 0 ? static_cast<int>(matrix_local[0].size()) : 0);

  // Безопасные отладочные выводы
  if (mpi_initialized && world_rank == 0) {
    std::cout << "MPI Process 0: Matrix " << rows_local << "x" << cols_local << std::endl;
    int real_max = std::numeric_limits<int>::min();
    for (const auto &row : matrix_local) {
      for (int v : row) {
        if (v > real_max) {
          real_max = v;
        }
      }
    }
    std::cout << "MPI Process 0: Real max in matrix = " << real_max << std::endl;
    if (rows_local >= 1 && cols_local >= 1) {
      std::cout << "MPI Process 0: Sample elements - matrix[0][0] = " << matrix_local[0][0];
      if (cols_local >= 2) {
        std::cout << ", matrix[0][1] = " << matrix_local[0][1];
      }
      if (rows_local >= 2) {
        std::cout << ", matrix[1][0] = " << matrix_local[1][0];
      }
      std::cout << std::endl;
    }
  }

  if (mpi_initialized) {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (mpi_initialized && world_rank == 1 && rows_local > 0 && cols_local > 0) {
    std::cout << "MPI Process 1: Sample elements - matrix[0][0] = " << matrix_local[0][0];
    if (cols_local >= 2) {
      std::cout << ", matrix[0][1] = " << matrix_local[0][1];
    }
    if (rows_local >= 2) {
      std::cout << ", matrix[1][0] = " << matrix_local[1][0];
    }
    std::cout << std::endl;
  }

  if (mpi_initialized) {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Каждый процесс ищет максимум по своим строкам (циклически).
  int local_max = std::numeric_limits<int>::min();
  for (int i = world_rank; i < rows_local; i += world_size) {
    for (int j = 0; j < cols_local; ++j) {
      if (matrix_local[i][j] > local_max) {
        local_max = matrix_local[i][j];
      }
    }
  }

  std::cout << "MPI Process " << world_rank << ": Local max = " << local_max << std::endl;

  int global_max = std::numeric_limits<int>::min();
  if (mpi_initialized) {
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  } else {
    global_max = local_max;
  }
  if (mpi_initialized && world_rank == 0) {
    std::cout << "MPI Process 0: Global max found = " << global_max << std::endl;
    int sequential_max = std::numeric_limits<int>::min();
    for (const auto &row : matrix_local) {
      for (int v : row) {
        if (v > sequential_max) {
          sequential_max = v;
        }
      }
    }
    std::cout << "MPI Process 0: Sequential verification = " << sequential_max << std::endl;
  }

  GetOutput() = global_max;
  return true;
}

// Убираем GenerateTestMatrix - матрица уже приходит из GetInput()
bool IvanovaPMaxMatrixMPI::PostProcessingImpl() {
  return true;
}
}  // namespace ivanova_p_max_matrix
