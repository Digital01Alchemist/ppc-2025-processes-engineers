#pragma once

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

class IvanovaPMultiplicationSparseMatricesCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IvanovaPMultiplicationSparseMatricesCrsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  void MultiplyLocalRows(const CRSMatrix &A, const CRSMatrix &B, int start_row, int end_row,
                         std::vector<double> &values, std::vector<int> &col_indices, std::vector<int> &row_ptr);
};

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
