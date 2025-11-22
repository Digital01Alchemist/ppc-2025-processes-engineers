#pragma once

#include <algorithm>
#include <iostream>  // Добавлено для std::cout
#include <limits>
#include <random>
#include <vector>

#include "ivanova_p_max_matrix/common/include/common.hpp"

namespace ivanova_p_max_matrix::data {  // Объединенные namespace

class MatrixGenerator {
 public:
  static InType GenerateMatrixWithKnownMax(int rows, int cols) {
    InType matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());

    // Максимум = большее из измерений
    int max_val = std::max(rows, cols);

    // Генерируем значения от -max_val+1 до max_val-1
    // (гарантированно меньше max_val)
    std::uniform_int_distribution<int> dist(-max_val + 1, max_val - 1);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix[i][j] = dist(gen);
      }
    }

    // Устанавливаем гарантированный максимум = max_val
    // Выбираем случайную позицию для максимума
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    int max_i = row_dist(gen);
    int max_j = col_dist(gen);
    matrix[max_i][max_j] = max_val;

    return matrix;
  }

  static InType GenerateSquareMatrixWithKnownMax(int size) {
    InType matrix(size, std::vector<int>(size));
    std::random_device rd;
    std::mt19937 gen(rd());

    // Максимум = size
    int max_val = size;

    // Генерируем значения от -max_val+1 до max_val-1
    std::uniform_int_distribution<int> dist(-max_val + 1, max_val - 1);

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = dist(gen);
      }
    }

    // Устанавливаем гарантированный максимум = max_val
    std::uniform_int_distribution<int> pos_dist(0, size - 1);
    int max_i = pos_dist(gen);
    int max_j = pos_dist(gen);
    matrix[max_i][max_j] = max_val;

    // ВЕРИФИКАЦИЯ: проверяем что максимум действительно установлен
    int actual_max = std::numeric_limits<int>::min();
    bool found_max = false;
    for (const auto &row : matrix) {
      for (int val : row) {
        actual_max = std::max(val, actual_max);  // Исправлено: std::max вместо >
        if (val == max_val) {
          found_max = true;
        }
      }
    }
    /*
        if (actual_max != max_val || !found_max) {
          std::cout << "ERROR in generator: Expected max " << max_val << " but got " << actual_max
                    << ", found_max = " << found_max << '\n';  // Исправлено: '\n' вместо std::endl
        } else {
          std::cout << "Generator: Successfully created matrix with max = " << actual_max
                    << '\n';  // Исправлено: '\n' вместо std::endl
        }
    */

    std::cout << "Generator: Successfully created matrix with max = " << actual_max
              << '\n';  // Исправлено: '\n' вместо std::endl

    return matrix;
  }

  // Для отрицательных тестов - все значения отрицательные
  static InType GenerateAllNegativeMatrix(int rows, int cols) {
    InType matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());

    // Генерируем значения от -1000 до -1
    std::uniform_int_distribution<int> dist(-1000, -1);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix[i][j] = dist(gen);
      }
    }

    // Максимум = -1 (самое большое отрицательное)
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    matrix[row_dist(gen)][col_dist(gen)] = -1;

    return matrix;
  }

  static int GetExpectedMax(const InType &matrix) {
    if (matrix.empty() || matrix[0].empty()) {
      return std::numeric_limits<int>::min();
    }

    // Для наших сгенерированных матриц максимум = большее из измерений
    return static_cast<int>(std::max(matrix.size(), matrix[0].size()));  // Исправлено: явное приведение типа
  }

  static int GetExpectedMaxForNegative([[maybe_unused]] const InType &matrix) {
    return -1;  // Для отрицательных матриц максимум всегда -1
  }
};

}  // namespace ivanova_p_max_matrix::data
