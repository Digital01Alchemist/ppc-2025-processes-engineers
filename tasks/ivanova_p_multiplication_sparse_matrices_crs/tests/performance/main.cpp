#include <gtest/gtest.h>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/mpi/include/ops_mpi.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

class IvanovaPMultiplicationSparseMatricesCrsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kMatrixSize_ = 100000;
  InType input_data_{};

  void SetUp() override {
    CRSMatrix A;
    CRSMatrix B;

    A.n = kMatrixSize_;
    B.n = kMatrixSize_;

    A.row_ptr.resize(kMatrixSize_ + 1);
    B.row_ptr.resize(kMatrixSize_ + 1);

    // -------- Matrix A --------
    // A(i,i) = 1, A(i,i+1) = 2
    int nnzA = 0;
    A.row_ptr[0] = 0;
    for (int i = 0; i < kMatrixSize_; i++) {
      A.col_indices.push_back(i);
      A.values.push_back(1.0);
      nnzA++;

      if (i + 1 < kMatrixSize_) {
        A.col_indices.push_back(i + 1);
        A.values.push_back(2.0);
        nnzA++;
      }

      A.row_ptr[i + 1] = nnzA;
    }

    // -------- Matrix B --------
    // B(i,i) = 3
    int nnzB = 0;
    B.row_ptr[0] = 0;
    for (int i = 0; i < kMatrixSize_; i++) {
      B.col_indices.push_back(i);
      B.values.push_back(3.0);
      nnzB++;
      B.row_ptr[i + 1] = nnzB;
    }

    input_data_ = std::make_tuple(A, B);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &values = output_data.values;
    const auto &col_indices = output_data.col_indices;
    const auto &row_ptr = output_data.row_ptr;

    // MPI: на не-root процессах результат может быть пустым
    if (row_ptr.empty()) {
      return true;
    }

    // CRS-инварианты
    if (row_ptr.front() != 0) {
      return false;
    }

    if (values.size() != col_indices.size()) {
      return false;
    }

    if (row_ptr.back() != static_cast<int>(values.size())) {
      return false;
    }

    // row_ptr монотонно неубывающий
    for (size_t i = 0; i + 1 < row_ptr.size(); i++) {
      if (row_ptr[i] > row_ptr[i + 1]) {
        return false;
      }
    }

    // корректность индексов столбцов
    for (int col : col_indices) {
      if (col < 0 || col >= output_data.n) {
        return false;
      }
    }

    return true;
  }
};

TEST_P(IvanovaPMultiplicationSparseMatricesCrsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IvanovaPMultiplicationSparseMatricesCrsMPI,
                                                       IvanovaPMultiplicationSparseMatricesCrsSEQ>(
    PPC_SETTINGS_ivanova_p_multiplication_sparse_matrices_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IvanovaPMultiplicationSparseMatricesCrsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IvanovaPMultiplicationSparseMatricesCrsPerfTests, kGtestValues, kPerfTestName);

}  // namespace ivanova_p_multiplication_sparse_matrices_crs
