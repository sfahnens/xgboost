#include "diff_dmatrix.h"

namespace xgboost {
namespace data {

void DiffMeta(MetaInfo const& a, MetaInfo const& b) {
  CHECK(a.num_row_ == b.num_row_) << "a=" << a.num_row_ << " b=" << b.num_row_;
  CHECK(a.num_col_ == b.num_col_) << "a=" << a.num_col_ << " b=" << b.num_col_;
  CHECK(a.num_nonzero_ == b.num_nonzero_)
      << "num_nonzero" << a.num_nonzero_ << " vs. " << b.num_nonzero_;
  CHECK(a.labels_.HostVector() == b.labels_.HostVector());
  CHECK(a.root_index_ == b.root_index_);
  CHECK(a.group_ptr_ == b.group_ptr_);
  CHECK(a.weights_.HostVector() == b.weights_.HostVector());
  CHECK(a.base_margin_.HostVector() == b.base_margin_.HostVector());
}

void DiffDMatrixByRowNotEmpty(DMatrix& a, DMatrix& b) {
  auto a_batch_set = a.GetRowBatches();
  auto b_batch_set = b.GetRowBatches();

  auto it_a = a_batch_set.begin();
  auto it_b = b_batch_set.begin();

  CHECK(it_a != a_batch_set.end()) << "a batch_set empty";
  CHECK(it_b != b_batch_set.end()) << "b batch_set empty";

  CHECK((*it_a).base_rowid == 0) << "it_a first base_rowid != 0";
  CHECK((*it_b).base_rowid == 0) << "it_b first base_rowid != 0";

  size_t batch_idx_a = 0;
  size_t batch_idx_b = 0;

  for (size_t curr_row = 0; curr_row < a.Info().num_row_; ++curr_row) {
    if (batch_idx_a >= (*it_a).Size()) {
      ++it_a;
      CHECK(it_a != a_batch_set.end()) << "a batch_set hit end";
      CHECK((*it_a).base_rowid == curr_row) << "id_a bad base_rowid";
      batch_idx_a = 0;
    }

    if (batch_idx_b >= (*it_b).Size()) {
      ++it_b;
      CHECK(it_a != a_batch_set.end()) << "b batch_set hit end";
      CHECK((*it_b).base_rowid == curr_row) << "it_b bad base_rowid";
      batch_idx_b = 0;
    }

    auto const& inst_a = (*it_a)[batch_idx_a];
    auto const& inst_b = (*it_b)[batch_idx_b];

    CHECK(inst_a.size() == inst_b.size()) << "row: inst size() mismatch";
    for (size_t i = 0; i < inst_a.size(); ++i) {
      CHECK(inst_a[i].index == inst_b[i].index) << "row: index mismatch";
      CHECK(inst_a[i].fvalue == inst_b[i].fvalue) << "row: fvalue mismatch";
    }

    ++batch_idx_a;
    ++batch_idx_b;
  }

  ++it_a;
  ++it_b;
  CHECK(it_a.AtEnd()) << "it_a not AtEnd()";
  CHECK(it_b.AtEnd()) << "it_b not AtEnd()";
  CHECK(!(it_a != a_batch_set.end())) << "a batch_set did not finish";
  CHECK(!(it_b != b_batch_set.end())) << "b batch_set did not finish";
}

void DiffDMatrix(DMatrix& a, DMatrix& b) {
  CHECK(&a != &b) << "cannot diff a matrix with itself";

  DiffMeta(a.Info(), b.Info());
  DiffDMatrixByRowNotEmpty(a, b);
}

}  // namespace data
}  // namespace xgboost
