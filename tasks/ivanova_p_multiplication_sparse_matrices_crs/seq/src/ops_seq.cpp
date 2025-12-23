#include "ivanova_p_multiplication_sparse_matrices_crs/seq/include/ops_seq.hpp"

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

IvanovaPMultiplicationSparseMatricesCrsSEQ::IvanovaPMultiplicationSparseMatricesCrsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return a.IsValid() && b.IsValid() && a.n == b.n;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::PreProcessingImpl() {
  GetOutput() = CRSMatrix{};
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::RunImpl() {
  const auto &[a, b] = GetInput();
  GetOutput() = MultiplyCRS(a, b);
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::PostProcessingImpl() {
  return GetOutput().IsValid();
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
