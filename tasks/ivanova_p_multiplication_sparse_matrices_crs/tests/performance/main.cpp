#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "ivanova_p_multiplication_sparse_matrices_crs/common/include/common.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/mpi/include/ops_mpi.hpp"
#include "ivanova_p_multiplication_sparse_matrices_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ivanova_p_multiplication_sparse_matrices_crs {

class IvanovaPMultiplicationSparseMatricesCrsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kMatrixSize = 100000;
  InType input_data;

  void SetUp() override {
    CRSMatrix a;
    CRSMatrix b;

    a.n = kMatrixSize;
    b.n = kMatrixSize;

    a.row_ptr.resize(static_cast<std::size_t>(kMatrixSize) + 1);
    b.row_ptr.resize(static_cast<std::size_t>(kMatrixSize) + 1);

    // -------- Matrix a --------
    // a(i,i) = 1, a(i,i+1) = 2
    int nnz_a = 0;
    a.row_ptr[0] = 0;
    for (int i = 0; i < kMatrixSize; i++) {
      a.col_indices.push_back(i);
      a.values.push_back(1.0);
      nnz_a++;

      if (i + 1 < kMatrixSize) {
        a.col_indices.push_back(i + 1);
        a.values.push_back(2.0);
        nnz_a++;
      }

      a.row_ptr[i + 1] = nnz_a;
    }

    // -------- Matrix b --------
    // b(i,i) = 3
    int nnz_b = 0;
    b.row_ptr[0] = 0;
    for (int i = 0; i < kMatrixSize; i++) {
      b.col_indices.push_back(i);
      b.values.push_back(3.0);
      nnz_b++;
      b.row_ptr[i + 1] = nnz_b;
    }

    input_data = std::make_tuple(a, b);
  }

  InType GetTestInputData() final {
    return input_data;
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

    const auto last_row_ptr = row_ptr.back();
    const auto values_size = static_cast<int>(values.size());
    if (last_row_ptr != values_size) {
      return false;
    }

    // row_ptr монотонно неубывающий
    for (std::size_t i = 0; i + 1 < row_ptr.size(); i++) {
      if (row_ptr[i] > row_ptr[i + 1]) {
        return false;
      }
    }

    // корректность индексов столбцов
    // ИСПРАВЛЕНИЕ: используем std::ranges::all_of вместо std::all_of с итераторами
    // clang-tidy требует использовать диапазонные алгоритмы (C++20)
    const bool all_cols_valid =
        std::ranges::all_of(col_indices,  // передаем контейнер напрямую, а не begin/end
                            [&output_data](int col) { return col >= 0 && col < output_data.n; });

    return all_cols_valid;
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
