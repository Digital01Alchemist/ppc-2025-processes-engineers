#include "ivanova_p_simple_iteration_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

namespace ivanova_p_simple_iteration_method {

namespace {

// Новая функция: один шаг метода простой итерации
void SimpleIterationStep(const std::vector<double> &A, const std::vector<double> &x, const std::vector<double> &b,
                         std::vector<double> &x_new, int n, double tau) {
  for (int i = 0; i < n; ++i) {
    double ax = 0.0;
    for (int j = 0; j < n; ++j) {
      ax += A[i * n + j] * x[j];
    }
    x_new[i] = x[i] - tau * (ax - b[i]);
  }
}

// Новая функция: вычисление нормы разности
double ComputeDiffNorm(const std::vector<double> &x, const std::vector<double> &x_new) {
  double diff = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double d = x_new[i] - x[i];
    diff += d * d;
  }
  return std::sqrt(diff);
}

}  // namespace

IvanovaPSimpleIterationMethodSEQ::IvanovaPSimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IvanovaPSimpleIterationMethodSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool IvanovaPSimpleIterationMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool IvanovaPSimpleIterationMethodSEQ::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  // Создаем тестовую систему: A = I (единичная матрица), b = (1, 1, ..., 1)
  // Используем vector<vector<double>> для матрицы - это медленнее

  // Новая версия A
  std::vector<double> A(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    A[i * n + i] = 1.0;
  }
  // std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  std::vector<double> b(n, 1.0);
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  // Старая инициализация A
  /*
  // Инициализация единичной матрицы
  for (int i = 0; i < n; ++i) {
    A[i][i] = 1.0;
  }*/

  // Параметры метода
  const double tau = 0.5;
  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  // Новая версия основного цикла
  for (int iter = 0; iter < max_iterations; ++iter) {
    SimpleIterationStep(A, x, b, x_new, n, tau);

    if (ComputeDiffNorm(x, x_new) < epsilon) {
      x = x_new;
      break;
    }

    x.swap(x_new);
  }

  // Метод простой итерации (старый)
  /* for (int iteration = 0; iteration < max_iterations; ++iteration) {
      for (int i = 0; i < n; ++i) {
        double ax_i = 0.0;
        for (int j = 0; j < n; ++j) {
          ax_i += A[i][j] * x[j];
        }
        x_new[i] = x[i] - tau * (ax_i - b[i]);
      }

      // Проверка сходимости
      double diff = 0.0;
      for (int i = 0; i < n; ++i) {
        double d = x_new[i] - x[i];
        diff += d * d;
      }
      diff = std::sqrt(diff);

      // Обновление решения
      x = x_new;

      if (diff < epsilon) {
        break;
      }
    }*/

  // Вычисление суммы компонент
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }

  GetOutput() = static_cast<int>(std::round(sum));

  return true;
}

bool IvanovaPSimpleIterationMethodSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace ivanova_p_simple_iteration_method
