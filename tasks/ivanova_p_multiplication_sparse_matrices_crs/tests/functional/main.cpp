#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <tuple>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/data/matrix_generators.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/mpi/include/ops_mpi.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

class IvanovaPMultiplicationSparseMatricesCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int test_id = std::get<0>(params);
    std::string test_name = std::get<1>(params);

    CRSMatrix a;
    CRSMatrix b;

    switch (test_id) {
      // === Базовые тесты с единичной матрицей ===
      case 1: {  // I * I = I (3x3)
        a = CreateIdentityMatrix(3);
        b = CreateIdentityMatrix(3);
        break;
      }

      case 3: {  // A * I = A
        a = CreateTridiagonalMatrix(5, 1.0, 2.0, 3.0);
        b = CreateIdentityMatrix(5);
        break;
      }

      // === Тесты с нулевой матрицей ===
      case 5: {  // A * 0 = 0
        a = CreateDiagonalMatrix(4, 5.0);
        b = CreateZeroMatrix(4);
        break;
      }
      case 7: {  // 0 * 0 = 0
        a = CreateZeroMatrix(3);
        b = CreateZeroMatrix(3);
        break;
      }

      // === Диагональные матрицы ===
      case 8: {  // D1 * D2 = D3 (диагонали перемножаются)
        a = CreateDiagonalMatrix(5, 2.0);
        b = CreateDiagonalMatrix(5, 3.0);
        break;
      }
      case 9: {  // Большая диагональная
        a = CreateDiagonalMatrix(50, 1.5);
        b = CreateDiagonalMatrix(50, 2.5);
        break;
      }

      // === Трёхдиагональные матрицы ===
      case 10: {  // Tri * Tri
        a = CreateTridiagonalMatrix(5, 1.0, 4.0, 1.0);
        b = CreateTridiagonalMatrix(5, 1.0, 4.0, 1.0);
        break;
      }
      case 12: {  // Большая трёхдиагональная
        a = CreateTridiagonalMatrix(20, 1.0, 2.0, 1.0);
        b = CreateTridiagonalMatrix(20, 0.5, 1.0, 0.5);
        break;
      }

      // === Треугольные матрицы ===
      case 13: {  // Upper * Upper
        a = CreateUpperTriangularMatrix(4, 1.0);
        b = CreateUpperTriangularMatrix(4, 1.0);
        break;
      }
      case 14: {  // Lower * Lower
        a = CreateLowerTriangularMatrix(4, 1.0);
        b = CreateLowerTriangularMatrix(4, 1.0);
        break;
      }
      case 15: {  // Upper * Lower
        a = CreateUpperTriangularMatrix(5, 1.0);
        b = CreateLowerTriangularMatrix(5, 1.0);
        break;
      }
      // === Матрицы с одним элементом ===
      case 17: {  // Один элемент в (0,0)
        a = CreateSingleElementMatrix(4, 0, 0, 5.0);
        b = CreateSingleElementMatrix(4, 0, 0, 3.0);
        break;
      }
      case 19: {  // Несовместимые позиции (результат = 0)
        a = CreateSingleElementMatrix(4, 0, 0, 5.0);
        b = CreateSingleElementMatrix(4, 1, 1, 3.0);
        break;
      }

      // === Матрицы с пустыми строками ===
      case 20: {  // Пустые строки * диагональ
        a = CreateMatrixWithEmptyRows(6);
        b = CreateDiagonalMatrix(6, 2.0);
        break;
      }
      // === Антидиагональные матрицы ===
      case 23: {  // Anti * Anti
        a = CreateAntiDiagonalMatrix(4, 1.0);
        b = CreateAntiDiagonalMatrix(4, 1.0);
        break;
      }

      // === Случайные разреженные матрицы ===
      case 27: {  // Низкая плотность 10%
        a = CreateRandomSparseMatrix(10, 0.1, 42);
        b = CreateRandomSparseMatrix(10, 0.1, 43);
        break;
      }
      case 29: {  // Высокая плотность 50%
        a = CreateRandomSparseMatrix(6, 0.5, 46);
        b = CreateRandomSparseMatrix(6, 0.5, 47);
        break;
      }
      // === Специальные значения ===
      case 31: {  // Отрицательные значения
        a = CreateDiagonalMatrix(5, -2.0);
        b = CreateDiagonalMatrix(5, -3.0);
        break;
      }
      case 32: {  // Смешанные знаки
        a = CreateTridiagonalMatrix(5, -1.0, 2.0, -1.0);
        b = CreateTridiagonalMatrix(5, 1.0, -2.0, 1.0);
        break;
      }
      // === Граничные размеры ===
      case 35: {  // Минимальный размер 1x1
        a = CreateDiagonalMatrix(1, 5.0);
        b = CreateDiagonalMatrix(1, 3.0);
        break;
      }
      case 36: {  // Размер 2x2
        a = CreateTridiagonalMatrix(2, 1.0, 2.0, 3.0);
        b = CreateTridiagonalMatrix(2, 4.0, 5.0, 6.0);
        break;
      }
      default: {
        // Fallback: диагональные матрицы
        a = CreateDiagonalMatrix(3, 1.0);
        b = CreateDiagonalMatrix(3, 1.0);
        break;
      }
    }

    input_data_ = std::make_tuple(a, b);
    expected_ = MultiplyCRS(a, b);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    // Безопасная проверка MPI
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);

    if (is_mpi_initialized) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;  // Не-root процессы MPI
      }
    }

    // SEQ или MPI root
    return output_data == expected_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(IvanovaPMultiplicationSparseMatricesCrsFuncTests, CRSxCRSMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParams = {
    // Базовые тесты
    std::make_tuple(1, "identity_3x3"),
    std::make_tuple(3, "A_times_identity"),
    std::make_tuple(5, "A_times_zero"),
    std::make_tuple(7, "zero_times_zero"),
    // Диагональные и трёхдиагональные
    std::make_tuple(8, "diagonal_small"),
    std::make_tuple(10, "tridiag_times_tridiag"),
    std::make_tuple(12, "tridiag_large"),
    // Треугольные матрицы
    std::make_tuple(13, "upper_times_upper"),
    std::make_tuple(14, "lower_times_lower"),
    std::make_tuple(15, "upper_times_lower"),
    // Особые случаи
    std::make_tuple(17, "single_element_corner"),
    std::make_tuple(19, "single_element_incompatible"),
    std::make_tuple(20, "empty_rows_times_diag"),
    std::make_tuple(23, "antidiag_times_antidiag"),
    // Случайные разреженные
    std::make_tuple(27, "random_sparse_10pct"),
    std::make_tuple(29, "random_sparse_50pct"),
    // Специальные значения
    std::make_tuple(31, "negative_values"),
    std::make_tuple(32, "mixed_signs"),
    // Граничные размеры
    std::make_tuple(35, "size_1x1"),
    std::make_tuple(36, "size_2x2"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IvanovaPMultiplicationSparseMatricesCrsMPI, InType>(
                                               kTestParams, PPC_SETTINGS_ivanova_p_multiplication_sparse_matrices_crs),
                                           ppc::util::AddFuncTask<IvanovaPMultiplicationSparseMatricesCrsSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_ivanova_p_multiplication_sparse_matrices_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = IvanovaPMultiplicationSparseMatricesCrsFuncTests::PrintFuncTestName<
    IvanovaPMultiplicationSparseMatricesCrsFuncTests>;

INSTANTIATE_TEST_SUITE_P(CRSMatrixTests, IvanovaPMultiplicationSparseMatricesCrsFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace ivanova_p_multiplication_sparse_matrices_crs
