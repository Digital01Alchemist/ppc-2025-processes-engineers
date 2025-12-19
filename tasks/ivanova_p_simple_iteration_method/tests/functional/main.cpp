#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "ivanova_p_simple_iteration_method/common/include/common.hpp"
#include "ivanova_p_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "ivanova_p_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_simple_iteration_method {

class IvanovaPSimpleIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

// Основные параметризованные тесты
TEST_P(IvanovaPSimpleIterationMethodFuncTests, SimpleIterationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "size_1"),   std::make_tuple(2, "size_2"),
                                             std::make_tuple(3, "size_3"),   std::make_tuple(5, "size_5"),
                                             std::make_tuple(7, "size_7"),   std::make_tuple(10, "size_10"),
                                             std::make_tuple(15, "size_15"), std::make_tuple(20, "size_20"),
                                             std::make_tuple(30, "size_30"), std::make_tuple(50, "size_50")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IvanovaPSimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ivanova_p_simple_iteration_method),
                                           ppc::util::AddFuncTask<IvanovaPSimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ivanova_p_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    IvanovaPSimpleIterationMethodFuncTests::PrintFuncTestName<IvanovaPSimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, IvanovaPSimpleIterationMethodFuncTests, kGtestValues, kFuncTestName);

// ============================================
// ТЕСТЫ КРАЕВЫХ СЛУЧАЕВ
// ============================================

// Тесты для последовательной версии (SEQ)
TEST(IvanovaPSimpleIterationMethodEdgeCases, InvalidInputZeroSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(0);
  EXPECT_FALSE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, InvalidInputNegativeSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(-5);
  EXPECT_FALSE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, ValidInputPositiveSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, PreProcessingSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_EQ(task.GetOutput(), 0);  // После PreProcessing результат должен быть 0
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, FullExecutionSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);  // Для размера 5 сумма должна быть 5
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, MinimalSizeSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);  // Для размера 1 сумма должна быть 1
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, LargeSizeSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(100);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 100);  // Для размера 100 сумма должна быть 100
}

// Тесты для MPI версии
TEST(IvanovaPSimpleIterationMethodEdgeCases, InvalidInputZeroMPI) {
  IvanovaPSimpleIterationMethodMPI task(0);
  EXPECT_FALSE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, InvalidInputNegativeMPI) {
  IvanovaPSimpleIterationMethodMPI task(-5);
  EXPECT_FALSE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, ValidInputPositiveMPI) {
  IvanovaPSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, PreProcessingMPI) {
  IvanovaPSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, FullExecutionMPI) {
  IvanovaPSimpleIterationMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 5);  // Для размера 5 сумма должна быть 5
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, MinimalSizeMPI) {
  IvanovaPSimpleIterationMethodMPI task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 1);  // Для размера 1 сумма должна быть 1
}

// Тесты на согласованность результатов
TEST(IvanovaPSimpleIterationMethodEdgeCases, ConsistencySEQvsMPISmall) {
  IvanovaPSimpleIterationMethodSEQ seq_task(3);
  IvanovaPSimpleIterationMethodMPI mpi_task(3);

  seq_task.Validation();
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  mpi_task.Validation();
  mpi_task.PreProcessing();
  mpi_task.Run();
  mpi_task.PostProcessing();

  EXPECT_EQ(seq_task.GetOutput(), mpi_task.GetOutput());
  EXPECT_EQ(seq_task.GetOutput(), 3);
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, ConsistencySEQvsMPIMedium) {
  IvanovaPSimpleIterationMethodSEQ seq_task(10);
  IvanovaPSimpleIterationMethodMPI mpi_task(10);

  seq_task.Validation();
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  mpi_task.Validation();
  mpi_task.PreProcessing();
  mpi_task.Run();
  mpi_task.PostProcessing();

  EXPECT_EQ(seq_task.GetOutput(), mpi_task.GetOutput());
  EXPECT_EQ(seq_task.GetOutput(), 10);
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, SingleProcessMPIEqualsSEQ) {
  // MPI с одним процессом должно работать как SEQ
  IvanovaPSimpleIterationMethodSEQ seq_task(7);
  IvanovaPSimpleIterationMethodMPI mpi_task(7);

  seq_task.Validation();
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  mpi_task.Validation();
  mpi_task.PreProcessing();
  mpi_task.Run();
  mpi_task.PostProcessing();

  EXPECT_EQ(seq_task.GetOutput(), mpi_task.GetOutput());
  EXPECT_EQ(seq_task.GetOutput(), 7);
}

// Тесты на правильность работы метода
TEST(IvanovaPSimpleIterationMethodEdgeCases, MethodConvergenceSmallSize) {
  // Проверяем, что метод сходится для малых размеров
  for (int n : {1, 2, 3, 4, 5}) {
    IvanovaPSimpleIterationMethodSEQ task(n);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), n);
  }
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, MethodConvergenceMediumSize) {
  // Проверяем, что метод сходится для средних размеров
  for (int n : {10, 20, 30}) {
    IvanovaPSimpleIterationMethodSEQ task(n);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), n);
  }
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, PostProcessingCheck) {
  // Проверяем PostProcessing - результат должен быть > 0
  IvanovaPSimpleIterationMethodSEQ task(8);
  task.Validation();
  task.PreProcessing();
  task.Run();
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, MultipleExecutionsSameTaskSEQ) {
  // Проверяем, что повторные вызовы Run дают тот же результат
  IvanovaPSimpleIterationMethodSEQ task(6);

  task.Validation();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  int first_result = task.GetOutput();

  // Повторный запуск
  task.Validation();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  int second_result = task.GetOutput();

  EXPECT_EQ(first_result, second_result);
  EXPECT_EQ(first_result, 6);
}

TEST(IvanovaPSimpleIterationMethodEdgeCases, MultipleExecutionsSameTaskMPI) {
  // Проверяем, что повторные вызовы Run дают тот же результат для MPI
  IvanovaPSimpleIterationMethodMPI task(6);

  task.Validation();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  int first_result = task.GetOutput();

  // Повторный запуск
  task.Validation();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  int second_result = task.GetOutput();

  EXPECT_EQ(first_result, second_result);
  EXPECT_EQ(first_result, 6);
}

// Тесты на граничные случаи с памятью и временем
TEST(IvanovaPSimpleIterationMethodEdgeCases, ModerateSizePerformanceCheck) {
  // Проверяем работу с умеренно большим размером
  IvanovaPSimpleIterationMethodSEQ task(200);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 200);
}

// Тесты на сохранение состояния
TEST(IvanovaPSimpleIterationMethodEdgeCases, StatePreservationSEQ) {
  IvanovaPSimpleIterationMethodSEQ task(4);

  // До Validation
  EXPECT_EQ(task.GetOutput(), 0);

  // После Validation
  EXPECT_TRUE(task.Validation());
  EXPECT_EQ(task.GetOutput(), 0);

  // После PreProcessing
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_EQ(task.GetOutput(), 0);

  // После Run
  EXPECT_TRUE(task.Run());
  EXPECT_NE(task.GetOutput(), 0);

  // После PostProcessing
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), 4);
}

}  // namespace
}  // namespace ivanova_p_simple_iteration_method
