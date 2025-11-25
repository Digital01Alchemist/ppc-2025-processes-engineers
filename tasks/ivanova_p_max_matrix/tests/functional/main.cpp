#include <gtest/gtest.h>

#include <algorithm>  // для std::max
#include <array>
#include <cstddef>   // для std::size_t
#include <iostream>  // для std::cout
#include <limits>    // для std::numeric_limits
#include <string>
#include <tuple>

#include "ivanova_p_max_matrix/common/include/common.hpp"
#include "ivanova_p_max_matrix/data/matrix_generator.hpp"
#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"
#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_max_matrix {

class IvanovaPMaxMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  IvanovaPMaxMatrixFuncTests() = default;

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_size_ = std::get<0>(params);
    matrix_type_ = std::get<1>(params);

    if (matrix_type_ == "negative") {
      test_matrix_ = data::MatrixGenerator::GenerateAllNegativeMatrix(matrix_size_, matrix_size_);
      expected_max_ = -1;
    } else {
      test_matrix_ = data::MatrixGenerator::GenerateSquareMatrixWithKnownMax(matrix_size_);
      expected_max_ = matrix_size_;
    }

    int actual_max = std::numeric_limits<int>::min();
    for (const auto &row : test_matrix_) {
      for (int val : row) {
        actual_max = std::max(val, actual_max);
      }
    }

    std::cout << "Test Setup: Matrix " << matrix_size_ << "x" << matrix_size_ << " - Expected max: " << expected_max_
              << ", Actual max: " << actual_max << '\n';
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = (expected_max_ == output_data);
    if (!result) {
      std::cout << "ERROR: Expected " << expected_max_ << " but got " << output_data << '\n';
    }
    return result;
  }

  InType GetTestInputData() final {
    return test_matrix_;
  }

 private:
  InType test_matrix_;
  int matrix_size_ = 0;
  std::string matrix_type_;
  int expected_max_ = 0;
};

namespace {

TEST_P(IvanovaPMaxMatrixFuncTests, FindMatrixMax) {
  ExecuteTest(GetParam());
}

class IvanovaPMaxMatrixSpecialTests : public ::testing::Test {
 protected:
  static void TestEmptyMatrix() {
    InType empty_matrix;
    IvanovaPMaxMatrixSEQ task(empty_matrix);
    EXPECT_FALSE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  static void TestJaggedMatrix() {
    InType jagged_matrix = {{1, 2}, {3}};
    IvanovaPMaxMatrixSEQ task(jagged_matrix);
    EXPECT_FALSE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  static void TestSingleElement() {
    InType single_element = {{42}};
    IvanovaPMaxMatrixSEQ task(single_element);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 42);
  }

  static void TestKnownMaxMatrix() {
    InType matrix = {{1, 2, 1}, {2, 1, 2}, {1, 3, 1}};
    IvanovaPMaxMatrixSEQ task(matrix);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 3);
  }

  // Новые тесты для покрытия конкретных непокрытых строк

  // Покрытие для seq строки 17: if (!in.empty())
  static void TestSeqNonEmptyConstructor() {
    InType non_empty_matrix = {{1, 2}, {3, 4}};
    IvanovaPMaxMatrixSEQ task(non_empty_matrix);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 4);
  }

  // Покрытие для seq строки 26: if (GetInput().empty())
  static void TestSeqEmptyInputValidation() {
    InType empty_matrix;
    IvanovaPMaxMatrixSEQ task(empty_matrix);
    // Эта строка уже покрыта в TestEmptyMatrix, но добавим явно
    EXPECT_FALSE(task.Validation());

    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  // Покрытие для mpi строки 46: if (GetInput().empty())
  static void TestMPIEmptyInputValidation() {
    InType empty_matrix;
    IvanovaPMaxMatrixMPI task(empty_matrix);
    EXPECT_FALSE(task.Validation());
    // Для MPI всегда вызываем полный цикл
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  // Покрытие для mpi строки 51: if (row.size() != cols)
  static void TestMPIDifferentRowSizes() {
    InType different_sizes_matrix = {{1, 2, 3},
                                     {4, 5},  // Разный размер - должно провалить валидацию
                                     {6, 7, 8}};
    IvanovaPMaxMatrixMPI task(different_sizes_matrix);
    EXPECT_FALSE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  // Покрытие для mpi строки 87: if (vec.empty())
  // Эта строка находится во вспомогательной функции CalculateDistribution
  // Покрываем через создание матрицы с разными размерами
  static void TestMPISingleRowMatrix() {
    InType single_row_matrix = {{1, 2, 3, 4, 5}};
    IvanovaPMaxMatrixMPI task(single_row_matrix);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 5);
  }

  // Дополнительный тест для покрытия случая с одной колонкой
  static void TestMPISingleColumnMatrix() {
    InType single_column_matrix = {{1}, {2}, {3}, {4}, {5}};
    IvanovaPMaxMatrixMPI task(single_column_matrix);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 5);
  }

  // Тест для очень маленькой матрицы (1x1)
  static void TestMPISingleElementMatrix() {
    InType single_element = {{42}};
    IvanovaPMaxMatrixMPI task(single_element);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 42);
  }

  static void TestMPIClassDirectCreation() {
    InType simple_matrix = {{1, 2}, {3, 4}};
    // Прямое создание объекта MPI класса
    IvanovaPMaxMatrixMPI mpi_task(simple_matrix);
    // Просто проверяем, что объект создался
    EXPECT_TRUE(mpi_task.Validation());
    EXPECT_TRUE(mpi_task.PreProcessing());
    EXPECT_TRUE(mpi_task.Run());
    EXPECT_TRUE(mpi_task.PostProcessing());
    EXPECT_EQ(mpi_task.GetStaticTypeOfTask(), ppc::task::TypeOfTask::kMPI);
  }

  static void TestSEQClassDirectCreation() {
    InType simple_matrix = {{1, 2}, {3, 4}};
    // Прямое создание объекта SEQ класса
    IvanovaPMaxMatrixSEQ seq_task(simple_matrix);
    // Просто проверяем, что объект создался
    EXPECT_TRUE(seq_task.Validation());
    EXPECT_TRUE(seq_task.PreProcessing());
    EXPECT_TRUE(seq_task.Run());
    EXPECT_TRUE(seq_task.PostProcessing());
    EXPECT_EQ(seq_task.GetStaticTypeOfTask(), ppc::task::TypeOfTask::kSEQ);
  }
};

// Существующие тесты
TEST_F(IvanovaPMaxMatrixSpecialTests, EmptyMatrix) {
  TestEmptyMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, JaggedMatrix) {
  TestJaggedMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, SingleElement) {
  TestSingleElement();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, KnownMaxMatrix) {
  TestKnownMaxMatrix();
}

// Новые тесты для покрытия конкретных строк
TEST_F(IvanovaPMaxMatrixSpecialTests, SeqNonEmptyConstructor) {
  TestSeqNonEmptyConstructor();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, SeqEmptyInputValidation) {
  TestSeqEmptyInputValidation();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPIEmptyInputValidation) {
  TestMPIEmptyInputValidation();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPIDifferentRowSizes) {
  TestMPIDifferentRowSizes();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPISingleRowMatrix) {
  TestMPISingleRowMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPISingleColumnMatrix) {
  TestMPISingleColumnMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPISingleElementMatrix) {
  TestMPISingleElementMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, MPIClassDirectCreation) {
  TestMPIClassDirectCreation();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, SEQClassDirectCreation) {
  TestSEQClassDirectCreation();
}

const std::array<TestType, 6> kTestMatrices = {std::make_tuple(10, "small"),    std::make_tuple(100, "medium"),
                                               std::make_tuple(500, "large"),   std::make_tuple(1000, "xlarge"),
                                               std::make_tuple(50, "negative"), std::make_tuple(128, "rectangular")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IvanovaPMaxMatrixMPI, InType>(kTestMatrices, PPC_SETTINGS_ivanova_p_max_matrix),
    ppc::util::AddFuncTask<IvanovaPMaxMatrixSEQ, InType>(kTestMatrices, PPC_SETTINGS_ivanova_p_max_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IvanovaPMaxMatrixFuncTests::PrintFuncTestName<IvanovaPMaxMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMaxTests, IvanovaPMaxMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ivanova_p_max_matrix
