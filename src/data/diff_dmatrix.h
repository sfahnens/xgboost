#pragma once

namespace xgboost {
namespace data {

inline void diff_meta(MetaInfo const& a, MetaInfo const& b) {
  CHECK(a.num_row == b.num_row) << "num_row mismatch";
  CHECK(a.num_col == b.num_col) << "num_col mismatch";
  CHECK(a.num_nonzero == b.num_nonzero) 
     << "num_nonzero mismatch" << a.num_nonzero << " vs. " << b.num_nonzero;
  CHECK(a.labels == b.labels) << "labels mismatch";
  CHECK(a.root_index == b.root_index) << "root_index mismatch";
  CHECK(a.group_ptr == b.group_ptr) << "group_ptr mismatch";
  CHECK(a.weights == b.weights) << "weights mismatch";
  CHECK(a.base_margin == b.base_margin) << "base_margin mismatch";
}

inline void diff_dmatrix_by_row(DMatrix& a, DMatrix& b) {
  auto* it_a = a.RowIterator();
  it_a->BeforeFirst();
  CHECK(it_a->Next()) << "it_a empty";
  CHECK(it_a->Value().base_rowid == 0) << "it_a first base_rowid != 0";

  auto* it_b = b.RowIterator();
  it_b->BeforeFirst();
  CHECK(it_b->Next()) << "it_b empty";
  CHECK(it_b->Value().base_rowid == 0) << "it_b first base_rowid != 0";

  size_t batch_idx_a = 0;
  size_t batch_idx_b = 0;

  for(size_t curr_row = 0; curr_row < a.info().num_row; ++curr_row) {
    if(batch_idx_a >= it_a->Value().size) {
      it_a->Next();
      CHECK(it_a->Value().base_rowid == curr_row) << "id_a bad base_rowid";
      batch_idx_a = 0;
    }
    if(batch_idx_b >= it_b->Value().size) {
      it_b->Next();
      CHECK(it_b->Value().base_rowid == curr_row) << "id_b bad base_rowid";
      batch_idx_b = 0;
    }

    SparseBatch::Inst const& inst_a = it_a->Value()[batch_idx_a];
    SparseBatch::Inst const& inst_b = it_b->Value()[batch_idx_b];

    CHECK(inst_a.length == inst_b.length) << "row: inst length mismatch";
    for(size_t i = 0; i < inst_a.length; ++i) {
      CHECK(inst_a[i].index == inst_b[i].index) << "row: index mismatch";
      CHECK(inst_a[i].fvalue == inst_b[i].fvalue) << "row: fvalue mismatch";
    }
  }
}

inline void diff_dmatrix(DMatrix& a, DMatrix& b) {
  CHECK(&a != &b) << "cannot diff a matrix with itself";

  diff_meta(a.info(), b.info());
  diff_dmatrix_by_row(a, b);
}

} // namespace xgboost
} // namespace xgboost
