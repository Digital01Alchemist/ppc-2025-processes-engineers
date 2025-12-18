#pragma once

#include "ivanova_p_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ivanova_p_simple_iteration_method {

class IvanovaPSimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IvanovaPSimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ivanova_p_simple_iteration_method
