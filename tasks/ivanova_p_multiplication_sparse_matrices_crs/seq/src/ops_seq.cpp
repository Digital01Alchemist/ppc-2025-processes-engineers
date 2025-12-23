#include "ivanova_p_multiplication_sparse_matrices_crs/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

IvanovaPMultiplicationSparseMatricesCrsSEQ::IvanovaPMultiplicationSparseMatricesCrsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::ValidationImpl() {
  const auto &[A, B] = GetInput();
  return A.IsValid() && B.IsValid() && A.n == B.n;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::PreProcessingImpl() {
  GetOutput() = CRSMatrix{};
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::RunImpl() {
  const auto &[A, B] = GetInput();
  GetOutput() = MultiplyCRS(A, B);
  return true;
}

bool IvanovaPMultiplicationSparseMatricesCrsSEQ::PostProcessingImpl() {
  return GetOutput().IsValid();
}

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
