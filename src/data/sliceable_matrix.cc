#include "sliceable_matrix.h"

namespace xgboost {
namespace data {

RowBatch make_row_batch(Slice const& slice, size_t offset) {
  RowBatch batch;
  batch.size = slice.row_count();
  batch.ind_ptr = dmlc::BeginPtr(slice.rows_.ptr_);
  batch.data_ptr = dmlc::BeginPtr(slice.rows_.data_);

  batch.base_rowid = offset;
  offset += slice.row_count();
  return batch;
}

Page make_column_page(std::vector<SparseBatch::Inst> const& rows,
                      size_t page_offset, size_t num_col) {
  const int nthread =
      std::min(omp_get_max_threads(), std::max(omp_get_num_procs() / 2 - 2, 1));
  Page p;
  common::ParallelGroupBuilder<SparseBatch::Entry> builder(&p.ptr_, &p.data_);
  builder.InitBudget(num_col, nthread);

#pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < rows.size(); ++i) {
    int tid = omp_get_thread_num();
    auto const& inst = rows[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      builder.AddBudget(inst[j].index, tid);
    }
  }
  builder.InitStorage();
#pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < rows.size(); ++i) {
    int tid = omp_get_thread_num();
    RowBatch::Inst inst = rows[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      auto const& e = inst[j];
      builder.Push(e.index, SparseBatch::Entry(i + page_offset, e.fvalue), tid);
    }
  }
// CHECK_EQ(pcol->Size(), info().num_col);

// sort columns
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
  for (bst_omp_uint i = 0; i < num_col; ++i) {
    if (p.ptr_[i] < p.ptr_[i + 1]) {
      std::sort(dmlc::BeginPtr(p.data_) + p.ptr_[i],
                dmlc::BeginPtr(p.data_) + p.ptr_[i + 1],
                SparseBatch::Entry::CmpValue);
    }
  }

  return p;
}

void csr_to_csc(Slice& slice, size_t offset) {
  auto batch = make_row_batch(slice, offset);
  std::vector<SparseBatch::Inst> rows;

  size_t col_page_offset = offset;
  auto finalize_col_page = [&] {
    slice.cols_.emplace_back(
        make_column_page(rows, col_page_offset, slice.info_.num_col));

    slice.col_offsets_.push_back(col_page_offset);
    slice.col_sizes_.push_back(rows.size());

    col_page_offset += rows.size();
  };

  for (size_t i = 0; i < slice.row_count(); ++i) {
    rows.emplace_back(batch[i]);

    if (rows.size() >= kMaxRowsPerBatch) {
      finalize_col_page();
      rows.clear();
    }
  }

  if (rows.size() > 0) {
    finalize_col_page();
  }
}

template <typename Vec>
void copy_slice(MetaInfo const& src, MetaInfo& dst,
                std::vector<size_t> const& indices, Vec vec) {
  if ((src.*vec).empty()) {
    return;
  }

  (dst.*vec).reserve(indices.size());
  for (auto const& i : indices) {
    (dst.*vec).push_back((src.*vec).at(i));
  }
}

void finalizes_slices(std::vector<Slice>& slices, MetaInfo const& nfo,
                      std::vector<std::vector<size_t>> const& indices) {
  verify(slices.size() == indices.size(), "slice/indices size mismatch");

  SliceConfigState config_state;
  for (auto i = 0ul; i < slices.size(); ++i) {
    config_state.set(i);
  }

  size_t offset = 0;

  for (auto i = 0ul; i < slices.size(); ++i) {
    auto& s = slices.at(i);
    s.config_state_ = config_state;

    auto const& idx = indices.at(i);
    verify(idx.size() == s.row_count(), "row size mismatch idx=%zu data=%zu",
           idx.size(), s.row_count());

    s.info_.num_row = s.row_count();
    s.info_.num_col = nfo.num_col;
    s.info_.num_nonzero = s.rows_.data_.size();

    copy_slice(nfo, s.info_, idx, &MetaInfo::labels);
    copy_slice(nfo, s.info_, idx, &MetaInfo::root_index);
    copy_slice(nfo, s.info_, idx, &MetaInfo::group_ptr);
    copy_slice(nfo, s.info_, idx, &MetaInfo::weights);
    copy_slice(nfo, s.info_, idx, &MetaInfo::base_margin);

    csr_to_csc(s, offset);

    offset += s.row_count();
  }
  verify(offset == nfo.num_row, "sum row size mismatch");
}

std::vector<Slice> make_slices(
    DMatrix* mat, std::vector<std::vector<size_t>> const& indices) {
  std::vector<Slice> slices;

  for (auto const& idx : indices) {
    size_t curr = 0;
    size_t idx_pos = 0;

    Page rows;

    dmlc::DataIter<RowBatch>* it = mat->RowIterator();
    it->BeforeFirst();
    while (it->Next()) {
      auto const& batch = it->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if(idx_pos >= idx.size()) {
          break;
        }
        if (curr++ < idx[idx_pos]) {
          continue;
        }
        ++idx_pos;

        RowBatch::Inst inst = batch[i];
        rows.data_.insert(rows.data_.end(), inst.data, inst.data + inst.length);
        rows.ptr_.push_back(rows.ptr_.back() + inst.length);
      }
    }
    verify(rows.ptr_.size() - 1 == idx.size(),
           "not all rows found! (rows=%zu, idx=%zu)", rows.ptr_.size() - 1,
           idx.size());

    slices.emplace_back(std::move(rows));
  }

  finalizes_slices(slices, mat->info(), indices);
  return slices;
}

bool ColBatchIter::Next() {
  if (slice_idx_ >= slices_.size()) {
    return false;
  }
  if (++page_idx_ >= slices_.at(slice_idx_)->cols_.size()) {
    ++slice_idx_;
    page_idx_ = 0;
  }
  if (slice_idx_ >= slices_.size()) {
    return false;
  }

  auto const& s = *slices_.at(slice_idx_);

  auto const& p = s.cols_.at(page_idx_);
  col_data_.resize(col_index_.size(), SparseBatch::Inst(NULL, 0));
  for (size_t i = 0; i < col_data_.size(); ++i) {
    col_data_[i] = p.get_inst(col_index_[i]);
  }

  batch_.size = col_index_.size();
  batch_.col_index = dmlc::BeginPtr(col_index_);
  batch_.col_data = dmlc::BeginPtr(col_data_);

  return true;
}

template <typename Field>
void merge_vector(MetaInfo& info, std::vector<Slice> const& slices,
                  std::vector<size_t> const& active, Field field) {
  if ((slices.at(active.at(0)).info_.*field).empty()) {
    return;
  }

  (info.*field).resize(info.num_row);
  size_t offset = 0;
  for (auto const& a : active) {
    auto const& s = slices.at(a);

    std::copy(begin(s.info_.*field), end(s.info_.*field),
              begin(info.*field) + offset);
    offset += s.info_.num_row;
  }
  verify((info.*field).size() == offset, "size mismatch");
}

SliceableMatrix::SliceableMatrix(std::shared_ptr<std::vector<Slice>> slices,
                                 std::vector<size_t> active)
    : slices_(std::move(slices)), active_(std::move(active)) {
  verify(std::all_of(begin(active_), end(active_),
                     [this](size_t a) { return a < slices_->size(); }),
         "invalid active slice");
  std::vector<Slice*> active_slices;

  size_t offset = 0;
  for (auto const& a : active_) {
    desired_config_state_.set(a);
    active_slices.push_back(&slices_->at(a));

    auto const& s = slices_->at(a);
    row_batches_.vec_.emplace_back(make_row_batch(s, offset));

    info_.num_row += s.info_.num_row;
    info_.num_col = s.info_.num_col;
    info_.num_nonzero += s.info_.num_nonzero;

    if (col_sizes_.empty()) {
      col_sizes_.resize(info_.num_col, 0ul);
    }
    verify(col_sizes_.size() == info_.num_col, "inconsistent column count");
    for (auto const& cp : s.cols_) {
      for (auto i = 0u; i < col_sizes_.size(); ++i) {
        col_sizes_[i] += cp.get_inst(i).length;
      }
    }
  }

  row_set_ = {info_.num_row};
  col_batches_.slices_ = std::move(active_slices);

  merge_vector(info_, *slices_, active_, &MetaInfo::labels);
  merge_vector(info_, *slices_, active_, &MetaInfo::root_index);
  merge_vector(info_, *slices_, active_, &MetaInfo::group_ptr);
  merge_vector(info_, *slices_, active_, &MetaInfo::weights);
  merge_vector(info_, *slices_, active_, &MetaInfo::base_margin);
}

MetaInfo& SliceableMatrix::info() { return info_; }
MetaInfo const& SliceableMatrix::info() const { return info_; }

dmlc::DataIter<RowBatch>* SliceableMatrix::RowIterator() {
  row_batches_.BeforeFirst();
  return &row_batches_;
}

void SliceableMatrix::maybe_reindex_column_pages() {
  bool okay = true;
  for (auto const& a : active_) {
    okay &= slices_->at(a).config_state_ == desired_config_state_;
  }
  if (okay) {
    return;
  }

  std::cout << "reindex column pages!" << std::endl;

  size_t offset = 0;
  for (auto const& a : active_) {
    auto& s = slices_->at(a);

    verify(s.cols_.size() == s.col_sizes_.size() &&
               s.cols_.size() == s.col_offsets_.size(),
           "column page corruption!");

    for (auto i = 0u; i < s.cols_.size(); ++i) {
      for (auto& entry : s.cols_[i].data_) {
        entry.index += -s.col_offsets_[i] + offset;  // subtract old, add new
      }
      offset += s.col_sizes_[i];
    }

    std::cout << "update config state!" << std::endl;
    s.config_state_ = desired_config_state_;
  }
}

dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator() {
  maybe_reindex_column_pages();

  size_t ncol = this->info().num_col;
  col_batches_.col_index_.resize(ncol);
  for (size_t i = 0; i < ncol; ++i) {
    col_batches_.col_index_[i] = static_cast<bst_uint>(i);
  }
  col_batches_.BeforeFirst();
  return &col_batches_;
}

dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator(
    const std::vector<bst_uint>& fset) {
  maybe_reindex_column_pages();

  size_t ncol = this->info().num_col;
  col_batches_.col_index_.resize(0);
  for (size_t i = 0; i < fset.size(); ++i) {
    if (fset[i] < ncol) {
      col_batches_.col_index_.push_back(fset[i]);
    }
  }
  col_batches_.BeforeFirst();
  return &col_batches_;
}

void SliceableMatrix::InitColAccess(const std::vector<bool>& enabled,
                                    float subsample, size_t max_row_perbatch) {
  verify(subsample == 1, "unsupported subsample");
}
bool SliceableMatrix::HaveColAccess() const { return true; }
bool SliceableMatrix::SingleColBlock() const {
  return slices_->size() == 1 && slices_->at(0).cols_.size() == 1;
}

size_t SliceableMatrix::GetColSize(size_t cidx) const {
  return col_sizes_.at(cidx);
}

float SliceableMatrix::GetColDensity(size_t cidx) const {
  size_t nmiss = row_set_.size() - col_sizes_[cidx];
  return 1.0f - (static_cast<float>(nmiss)) / row_set_.size();
}

const RowSet& SliceableMatrix::buffered_rowset() const { return row_set_; };

}  // namespace data
}  // namespace xgboost
