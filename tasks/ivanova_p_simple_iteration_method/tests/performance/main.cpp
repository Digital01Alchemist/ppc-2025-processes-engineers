#include <gtest/gtest.h>

#include "ivanova_p_simple_iteration_method/common/include/common.hpp"
#include "ivanova_p_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "ivanova_p_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ivanova_p_simple_iteration_method {

class IvanovaPSimpleIterationMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2500;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(IvanovaPSimpleIterationMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IvanovaPSimpleIterationMethodMPI, IvanovaPSimpleIterationMethodSEQ>(
        PPC_SETTINGS_ivanova_p_simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IvanovaPSimpleIterationMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IvanovaPSimpleIterationMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace ivanova_p_simple_iteration_method
