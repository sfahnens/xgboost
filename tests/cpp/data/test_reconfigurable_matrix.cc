// Copyright by Contributors
#include <random>

#include "xgboost/c_api.h"
#include "xgboost/data.h"

#include "../../../src/data/simple_csr_source.h"
#include "../../../src/data/simple_dmatrix.h"

#include "../../../src/data/diff_dmatrix.h"
#include "../../../src/data/reconfigurable_matrix.h"

#include "../helpers.h"

using namespace xgboost::data;

TEST(ReconfigurableMatrix, MatrixToSlices) {
  auto* smat = xgboost::CreateDMatrix(2, 2, 0.1);

  auto src = ReconfigurableSource::Create(smat->get(), {{0}, {1}});

  ASSERT_EQ(src->batches_.size(), 2);
  ReconfigurableMatrix rmat{src, {0, 1}};

  DiffMeta((**smat).Info(), rmat.Info());
  DiffDMatrixByRowNotEmpty(**smat, rmat);

  delete smat;
}

TEST(ReconfigurableMatrix, Copy) {
  auto* smat = xgboost::CreateDMatrix(2, 10, 0.1);

  auto src = ReconfigurableSource::Create(smat->get(), {{0}, {1}});
  ReconfigurableMatrix rmat{src, {0, 1}};

  auto copy_src = std::unique_ptr<SimpleCSRSource>(new SimpleCSRSource());
  copy_src->CopyFrom(&rmat);
  SimpleDMatrix copy_mat{std::move(copy_src)};

  DiffMeta((**smat).Info(), rmat.Info());
  DiffDMatrixByRowNotEmpty(**smat, rmat);

  delete smat;
}

TEST(ReconfigurableMatrix, ColAccess) {
  auto* smat = xgboost::CreateDMatrix(2, 10, 0.1);

  auto src = ReconfigurableSource::Create(smat->get(), {{0}, {1}});
  ReconfigurableMatrix rmat{src, {0, 1}};

  DiffMeta((**smat).Info(), rmat.Info());

  for (auto i = 0; i < rmat.Info().num_col_; ++i) {
    EXPECT_EQ((**smat).GetColDensity(i), rmat.GetColDensity(i));
  }

  auto num_col_batch = 0;
  for (auto& page : rmat.GetSortedColumnBatches()) {
    num_col_batch += 1;
    EXPECT_EQ(page.Size(), rmat.Info().num_col_)
        << "Expected batch size = num_cols as max_row_perbatch is 1.";
    for (int i = 0; i < static_cast<int>(page.Size()); ++i) {
      EXPECT_LE(page[i].size(), 1)
          << "Expected length of each col <=1 as max_row_perbatch is 1. ";
    }
  }

  EXPECT_EQ(num_col_batch, rmat.Info().num_row_)
      << "Expected num batches = num_rows as max_row_perbatch is 1";

  delete smat;
}

void compare_active(ReconfigurableMatrix& act, ReconfigurableMatrix& exp) {
  std::vector<int> act_idx;
  for (auto const& page : act.GetSortedColumnBatches()) {
    auto const it = std::find_if(
        begin(act.source_->batches_), end(act.source_->batches_),
        [&](ReconfigurableBatch const& s) { return &s.cols_ == &page; });
    ASSERT_NE(it, end(act.source_->batches_));
    act_idx.emplace_back(std::distance(begin(act.source_->batches_), it));
  }

  std::vector<int> exp_idx;
  for (auto const& page : exp.GetSortedColumnBatches()) {
    auto const it = std::find_if(
        begin(exp.source_->batches_), end(exp.source_->batches_),
        [&](ReconfigurableBatch const& s) { return &s.cols_ == &page; });
    ASSERT_NE(it, end(exp.source_->batches_));
    exp_idx.emplace_back(std::distance(begin(exp.source_->batches_), it));
  }

  ASSERT_EQ(act_idx, exp_idx);
}

template <typename T, typename Fn>
void compare_vec(std::vector<T> const& actual, std::vector<T> const& expected,
                 Fn&& fn) {
  ASSERT_EQ(actual.size(), expected.size());
  for (auto i = 0u; i < actual.size(); ++i) {
    fn(actual.at(i), expected.at(i));
  }
}

void compare_slices(std::vector<ReconfigurableBatch> const& actual,
                    std::vector<ReconfigurableBatch> const& expected) {
  compare_vec(
      actual, expected,
      [](ReconfigurableBatch const& sa, ReconfigurableBatch const& se) {
        EXPECT_EQ(sa.config_state_, se.config_state_);

        EXPECT_EQ(sa.info_.num_row_, se.info_.num_row_);
        EXPECT_EQ(sa.info_.num_col_, se.info_.num_col_);
        EXPECT_EQ(sa.info_.num_nonzero_, se.info_.num_nonzero_);
        EXPECT_EQ(sa.info_.labels_.HostVector(), se.info_.labels_.HostVector());
        EXPECT_EQ(sa.info_.root_index_, se.info_.root_index_);
        EXPECT_EQ(sa.info_.group_ptr_, se.info_.group_ptr_);
        EXPECT_EQ(sa.info_.weights_.HostVector(),
                  se.info_.weights_.HostVector());
        EXPECT_EQ(sa.info_.base_margin_.HostVector(),
                  se.info_.base_margin_.HostVector());

        EXPECT_EQ(sa.rows_.base_rowid, se.rows_.base_rowid);
        EXPECT_EQ(sa.rows_.offset.HostVector(), se.rows_.offset.HostVector());
        EXPECT_EQ(sa.rows_.data.HostVector(), se.rows_.data.HostVector());

        EXPECT_EQ(sa.cols_.base_rowid, se.cols_.base_rowid);
        EXPECT_EQ(sa.cols_.offset.HostVector(), se.cols_.offset.HostVector());
        EXPECT_EQ(sa.cols_.data.HostVector(), se.cols_.data.HostVector());
      });
}

TEST(ReconfigurableMatrix, Reindex) {
  auto* dmat = xgboost::CreateDMatrix(20, 100, 0.5);

  std::vector<std::vector<size_t>> indices{3};

  std::uniform_int_distribution<> dist{0, static_cast<int>(indices.size()) - 1};
  std::mt19937 gen;
  for (auto i = 0u; i < (**dmat).Info().num_row_; ++i) {
    indices[dist(gen)].push_back(i);
  }

  auto s_ref = ReconfigurableSource::Create(dmat->get(), indices);
  auto s_test = ReconfigurableSource::Create(dmat->get(), indices);

  // pristine (all inactive / offset zero)
  compare_slices(s_ref->batches_, s_test->batches_);

  // no index happens -> lazy
  ReconfigurableMatrix smat_learn_ref{s_ref, {0, 1}};
  ReconfigurableMatrix smat_learn_test_a{s_test, {0, 1}};
  compare_slices(s_ref->batches_, s_test->batches_);

  // compare_active triggers initial index
  compare_active(smat_learn_ref, smat_learn_test_a);
  compare_slices(s_ref->batches_, s_test->batches_);

  // trigger reindex
  ReconfigurableMatrix smat_learn_test_b{s_test, {1, 2}};
  smat_learn_test_b.GetSortedColumnBatches();

  // set back
  smat_learn_test_a.GetSortedColumnBatches();

  // slice 2 was changed -> reset it on both sides
  ReconfigurableMatrix smat_val_ref{s_ref, {2}};
  ReconfigurableMatrix smat_val_test_a{s_test, {2}};

  compare_active(smat_val_ref, smat_val_test_a);  // triggers reindex

  auto const& rs = s_ref->batches_;
  EXPECT_EQ(rs.at(0).rows_.base_rowid, 0);                  // smat_learn_ref
  EXPECT_EQ(rs.at(1).rows_.base_rowid, indices[0].size());  // smat_learn_ref
  EXPECT_EQ(rs.at(2).rows_.base_rowid, 0);                  // smat_val_ref

  auto const& ts = s_test->batches_;
  EXPECT_EQ(ts.at(0).rows_.base_rowid, 0);  // smat_learn_test_a
  EXPECT_EQ(ts.at(1).rows_.base_rowid, 0);  // smat_learn_test_b
  EXPECT_EQ(ts.at(2).rows_.base_rowid, 0);  // smat_val_test_a
}
