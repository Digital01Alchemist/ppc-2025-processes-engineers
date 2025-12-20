#include "ivanova_p_simple_iteration_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "ivanova_p_simple_iteration_method/common/include/common.hpp"

namespace ivanova_p_simple_iteration_method {

namespace {  // Анонимный namespace для вспомогательных функций

// Новая функция: один шаг метода простой итерации
void SimpleIterationStep(const std::vector<double> &a, const std::vector<double> &x, const std::vector<double> &b,
                         std::vector<double> &x_new, int n, double tau) {
  for (int i = 0; i < n; ++i) {
    double ax = 0.0;
    for (int j = 0; j < n; ++j) {
      ax += a[(static_cast<size_t>(i) * n) + j] * x[j];
    }
    x_new[i] = x[i] - (tau * (ax - b[i]));
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
  const size_t matrix_size = static_cast<size_t>(n) * n;
  std::vector<double> a(matrix_size, 0.0);
  for (int i = 0; i < n; ++i) {
    a[(static_cast<size_t>(i) * n) + i] = 1.0;
  }

  std::vector<double> b(n, 1.0);
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  // Параметры метода
  const double tau = 0.5;
  const double epsilon = 1e-6;
  const int max_iterations = 1000;

  // Метод простой итерации
  for (int iter = 0; iter < max_iterations; ++iter) {
    SimpleIterationStep(a, x, b, x_new, n, tau);

    if (ComputeDiffNorm(x, x_new) < epsilon) {
      x = x_new;
      break;
    }

    x.swap(x_new);
  }

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
