#pragma once

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

class IvanovaPMultiplicationSparseMatricesCrsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IvanovaPMultiplicationSparseMatricesCrsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
