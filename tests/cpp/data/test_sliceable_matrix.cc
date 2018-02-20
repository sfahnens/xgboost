// Copyright by Contributors
#include <random>

#include <xgboost/data.h>
#include "../../../src/data/simple_csr_source.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/data/sliceable_matrix.h"

#include "../helpers.h"

using namespace xgboost::data;

TEST(SliceableMatrix, MatrixToSlices) {
  std::string tmp_file = CreateSimpleTestData();
  auto const dmat = std::unique_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Load(tmp_file, true, false)};
  std::remove(tmp_file.c_str());

  // auto slices =
  //     std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), 1));
  auto slices =
      std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), {{0}, {1}}));

  ASSERT_EQ(slices->size(), 2);
  SliceableMatrix smat{slices, {0, 1}};

  auto* row_it = dmat->RowIterator();
  row_it->BeforeFirst();
  row_it->Next();

  for (auto row_idx = 0ul; row_idx < 2; ++row_idx) {
    auto const& orig_row = row_it->Value()[row_idx];

    auto const& slice_row = smat.row_batches_.vec_.at(row_idx)[0];
    ASSERT_EQ(orig_row.length, slice_row.length);

    for (auto i = 0; i < orig_row.length; ++i) {
      EXPECT_EQ(orig_row[i].index, slice_row[i].index);
      EXPECT_EQ(orig_row[i].fvalue, slice_row[i].fvalue);
    }
  }
}

TEST(SliceableMatrix, Copy) {
  std::string tmp_file = CreateSimpleTestData();
  auto const dmat = std::unique_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Load(tmp_file, true, false)};
  std::remove(tmp_file.c_str());

  // auto slices =
  //     std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), 1));
  auto slices =
      std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), {{0}, {1}}));
  SliceableMatrix smat{slices, {0, 1}};

  auto copy_src = std::unique_ptr<SimpleCSRSource>(new SimpleCSRSource());
  copy_src->CopyFrom(&smat);
  SimpleDMatrix copy_mat{std::move(copy_src)};

  EXPECT_EQ(dmat->info().num_col, copy_mat.info().num_col);
  EXPECT_EQ(dmat->info().num_row, copy_mat.info().num_row);
  EXPECT_EQ(dmat->info().num_nonzero, copy_mat.info().num_nonzero);
  EXPECT_EQ(dmat->info().labels, copy_mat.info().labels);
  EXPECT_EQ(dmat->info().root_index, copy_mat.info().root_index);
  EXPECT_EQ(dmat->info().group_ptr, copy_mat.info().group_ptr);
  EXPECT_EQ(dmat->info().weights, copy_mat.info().weights);
  EXPECT_EQ(dmat->info().base_margin, copy_mat.info().base_margin);

  auto* orig_it = dmat->RowIterator();
  orig_it->BeforeFirst();
  ASSERT_TRUE(orig_it->Next());

  auto* copy_it = copy_mat.RowIterator();
  copy_it->BeforeFirst();
  ASSERT_TRUE(copy_it->Next());

  ASSERT_EQ(orig_it->Value().size, copy_it->Value().size);
  for (auto row_idx = 0ul; row_idx < orig_it->Value().size; ++row_idx) {
    auto const& orig_row = orig_it->Value()[row_idx];
    auto const& copy_row = copy_it->Value()[row_idx];

    ASSERT_EQ(orig_row.length, copy_row.length);
    for (auto i = 0; i < orig_row.length; ++i) {
      EXPECT_EQ(orig_row[i].index, copy_row[i].index);
      EXPECT_EQ(orig_row[i].fvalue, copy_row[i].fvalue);
    }
  }

  ASSERT_FALSE(orig_it->Next());
  ASSERT_FALSE(copy_it->Next());
}

TEST(SliceableMatrix, ColAccess) {
  std::string tmp_file = CreateSimpleTestData();
  auto const dmat = std::unique_ptr<xgboost::DMatrix>{
      xgboost::DMatrix::Load(tmp_file, true, false)};
  std::remove(tmp_file.c_str());

  // auto slices =
  //     std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), 1));
  auto slices =
      std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), {{0}, {1}}));
  SliceableMatrix smat{slices, {0, 1}};

  ASSERT_TRUE(smat.HaveColAccess());

  EXPECT_EQ(smat.GetColSize(0), 2);
  EXPECT_EQ(smat.GetColSize(1), 1);
  EXPECT_EQ(smat.GetColDensity(0), 1);
  EXPECT_EQ(smat.GetColDensity(1), 0.5);

  dmlc::DataIter<xgboost::ColBatch>* col_iter = smat.ColIterator();

  long num_col_batch = 0;
  col_iter->BeforeFirst();
  while (col_iter->Next()) {
    num_col_batch += 1;
    EXPECT_EQ(col_iter->Value().size, dmat->info().num_col)
        << "Expected batch size = num_cols as max_row_perbatch is 1.";
    for (int i = 0; i < static_cast<int>(col_iter->Value().size); ++i) {
      EXPECT_LE(col_iter->Value()[i].length, 1)
          << "Expected length of each colbatch <=1 as max_row_perbatch is 1.";
    }
  }
  EXPECT_EQ(num_col_batch, dmat->info().num_row)
      << "Expected num batches = num_rows as max_row_perbatch is 1";
}

template <typename T, typename Fn>
void compare_vec(std::vector<T> const& actual, std::vector<T> const& expected,
                 Fn&& fn) {
  ASSERT_EQ(actual.size(), expected.size());
  for (auto i = 0u; i < actual.size(); ++i) {
    auto const& sa = actual.at(i);
    auto const& se = expected.at(i);

    fn(sa, se);
  }
}

void compare_page(Page const& actual, Page const& expected) {
  EXPECT_EQ(actual.ptr_, expected.ptr_);

  compare_vec(actual.data_, expected.data_,
              [](xgboost::SparseBatch::Entry const& a,
                 xgboost::SparseBatch::Entry const& e) {
                EXPECT_EQ(a.index, e.index);
                EXPECT_EQ(a.fvalue, e.fvalue);
              });
}

void compare_slices(std::vector<Slice> const& actual,
                    std::vector<Slice> const& expected) {
  compare_vec(actual, expected, [](Slice const& sa, Slice const& se) {
    EXPECT_EQ(sa.config_state_, se.config_state_);

    EXPECT_EQ(sa.info_.num_row, se.info_.num_row);
    EXPECT_EQ(sa.info_.num_col, se.info_.num_col);
    EXPECT_EQ(sa.info_.num_nonzero, se.info_.num_nonzero);
    EXPECT_EQ(sa.info_.labels, se.info_.labels);
    EXPECT_EQ(sa.info_.root_index, se.info_.root_index);
    EXPECT_EQ(sa.info_.group_ptr, se.info_.group_ptr);
    EXPECT_EQ(sa.info_.weights, se.info_.weights);
    EXPECT_EQ(sa.info_.base_margin, se.info_.base_margin);

    compare_page(sa.rows_, se.rows_);

    ASSERT_EQ(sa.col_offsets_, se.col_offsets_);
    ASSERT_EQ(sa.col_sizes_, se.col_sizes_);
    compare_vec(sa.cols_, se.cols_, &compare_page);
  });
}

TEST(SliceableMatrix, Reindex) {
  auto dmat = CreateDMatrix(20, 100, 0.5);

  std::vector<std::vector<size_t>> indices{3};

  std::uniform_int_distribution<> dist{0, static_cast<int>(indices.size())-1};
  std::mt19937 gen;

  for (auto i = 0u; i < dmat->info().num_row; ++i) {
    indices[dist(gen)].push_back(i);
  }

  auto s_ref =
      std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), indices));

  auto s_test =
      std::make_shared<std::vector<Slice>>(make_slices(dmat.get(), indices));

  compare_slices(*s_ref, *s_test);

  SliceableMatrix smat_learn_ref{s_ref, {0, 1}};

  SliceableMatrix smat_learn_test_a{s_test, {0, 1}};
  compare_slices(*s_ref, *s_test);

  smat_learn_ref.ColIterator();
  smat_learn_test_a.ColIterator();

  compare_slices(*s_ref, *s_test);

  // trigger reindex
  SliceableMatrix smat_learn_test_b{s_test, {1, 2}};
  smat_learn_test_b.ColIterator();

  // set back
  smat_learn_test_a.ColIterator();

  // slice 2 was changed -> reset it on both sides
  SliceableMatrix smat_val_ref{s_ref, {2}};
  SliceableMatrix smat_val_test_a{s_test, {2}};

  smat_val_ref.ColIterator();
  smat_val_test_a.ColIterator();

  compare_slices(*s_ref, *s_test);
}
