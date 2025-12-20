#include "ivanova_p_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "ivanova_p_simple_iteration_method/common/include/common.hpp"

namespace ivanova_p_simple_iteration_method {

namespace {  // Анонимный namespace для всех вспомогательных функций

// Распределение строк матрицы по процессам
void ComputeDistribution(int n, int size, std::vector<int> &row_counts, std::vector<int> &row_displs,
                         std::vector<int> &matrix_counts, std::vector<int> &matrix_displs) {
  int row_offset = 0;
  int matrix_offset = 0;

  for (int proc = 0; proc < size; ++proc) {
    int base_rows = n / size;
    int extra = (proc < (n % size)) ? 1 : 0;
    int proc_rows = base_rows + extra;

    row_counts[proc] = proc_rows;
    row_displs[proc] = row_offset;
    matrix_counts[proc] = proc_rows * n;
    matrix_displs[proc] = matrix_offset;

    row_offset += proc_rows;
    matrix_offset += proc_rows * n;
  }
}

// Инициализация матрицы и вектора (на нулевом процессе)
void InitializeSystem(std::vector<double> &flat_matrix, std::vector<double> &b, int n) {
  flat_matrix.resize(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    flat_matrix[(static_cast<size_t>(i) * n) + i] = 1.0;  // Единичная матрица
  }
  b.resize(n, 1.0);  // Вектор из единиц
}

// Вычисление локального произведения (часть матрично-векторного умножения)
void ComputeLocalProduct(const std::vector<double> &local_matrix, const std::vector<double> &x,
                         const std::vector<double> &local_b, std::vector<double> &local_x_new, int local_rows,
                         int start_row, int n, double tau) {
  for (int i = 0; i < local_rows; ++i) {
    double ax_i = 0.0;
    for (int j = 0; j < n; ++j) {
      ax_i += local_matrix[(static_cast<size_t>(i) * n) + j] * x[j];
    }
    local_x_new[i] = x[start_row + i] - (tau * (ax_i - local_b[i]));
  }
}

// Новая функция: сбор нового вектора
void AllGatherVector(const std::vector<double> &local_x, std::vector<double> &x_global,
                     const std::vector<int> &row_counts, const std::vector<int> &row_displs) {
  const int local_size = static_cast<int>(local_x.size());
  MPI_Allgatherv(local_x.data(), local_size, MPI_DOUBLE, x_global.data(), row_counts.data(), row_displs.data(),
                 MPI_DOUBLE, MPI_COMM_WORLD);
}

// Новая функция: проверка сходимости для всех процессов
bool CheckConvergenceAll(double local_diff, double epsilon) {
  double global_diff = 0.0;
  MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(global_diff) < epsilon;
}

// Вычисление локальной нормы разности
double ComputeLocalDiff(const std::vector<double> &x_new, const std::vector<double> &x, int local_rows, int start_row) {
  double local_diff = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    double d = x_new[start_row + i] - x[start_row + i];
    local_diff += d * d;
  }
  return local_diff;
}

}  // namespace

IvanovaPSimpleIterationMethodMPI::IvanovaPSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IvanovaPSimpleIterationMethodMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;
  if (rank == 0) {
    is_valid = ((GetInput() > 0) && (GetOutput() == 0)) ? 1 : 0;
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool IvanovaPSimpleIterationMethodMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IvanovaPSimpleIterationMethodMPI::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Вычисляем распределение данных
  std::vector<int> row_counts(size);
  std::vector<int> row_displs(size);
  std::vector<int> matrix_counts(size);
  std::vector<int> matrix_displs(size);

  ComputeDistribution(n, size, row_counts, row_displs, matrix_counts, matrix_displs);

  int local_rows = row_counts[rank];
  int start_row = row_displs[rank];

  // Подготовка данных на нулевом процессе
  std::vector<double> flat_matrix;
  std::vector<double> b;

  if (rank == 0) {
    InitializeSystem(flat_matrix, b, n);
  }

  // Распределение матрицы по процессам (плоский формат)
  std::vector<double> local_matrix(static_cast<size_t>(local_rows) * n, 0.0);
  MPI_Scatterv(flat_matrix.data(), matrix_counts.data(), matrix_displs.data(), MPI_DOUBLE, local_matrix.data(),
               local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распределение вектора b по процессам
  std::vector<double> local_b(local_rows, 0.0);
  MPI_Scatterv(b.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, local_b.data(), local_rows, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  // Параметры метода
  const double tau = 0.5;
  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  // Инициализация вектора решения
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(local_rows, 0.0);

  // Метод простой итерации
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    ComputeLocalProduct(local_matrix, x, local_b, local_x_new, local_rows, start_row, n, tau);

    AllGatherVector(local_x_new, x_new, row_counts, row_displs);

    double local_diff = ComputeLocalDiff(x_new, x, local_rows, start_row);

    x.swap(x_new);

    if (CheckConvergenceAll(local_diff, epsilon)) {
      break;
    }
  }

  // Вычисление суммы компонент вектора решения
  double local_sum = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    local_sum += x[start_row + i];
  }

  // Сбор результата на нулевом процессе
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Сохранение результата и рассылка всем процессам
  if (rank == 0) {
    GetOutput() = static_cast<int>(std::round(global_sum));
  }

  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool IvanovaPSimpleIterationMethodMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace ivanova_p_simple_iteration_method
