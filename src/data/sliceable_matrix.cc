#include "sliceable_matrix.h"

namespace xgboost {
namespace data {

template <typename T>
std::vector<T> copy_slice(std::vector<T> const& src, size_t idx, size_t count) {
  if (src.size() != 0) {
    return {begin(src) + idx, begin(src) + idx + count};
  } else {
    return {};
  }
}

std::vector<Slice> matrix_to_slices(DMatrix* mat, size_t nrow) {
  std::vector<Slice> slices;
  slices.emplace_back();

  dmlc::DataIter<RowBatch>* it = mat->RowIterator();
  it->BeforeFirst();
  while (it->Next()) {
    auto const& batch = it->Value();
    for (size_t i = 0; i < batch.size; ++i) {
      auto& s = slices.back();
      RowBatch::Inst inst = batch[i];
      s.row_data_.insert(s.row_data_.end(), inst.data, inst.data + inst.length);
      s.row_ptr_.push_back(s.row_ptr_.back() + inst.length);

      if (s.row_ptr_.size() - 1 == nrow) {
        slices.emplace_back();
      }
    }
  }
  if (slices.back().row_ptr_.size() == 1) {
    slices.erase(end(slices) - 1);
  }

  auto const& nfo = mat->info();

  size_t offset = 0;
  for (auto& s : slices) {
    auto const rows = s.row_count();

    s.info_.num_row = rows;
    s.info_.num_col = nfo.num_col;
    s.info_.num_nonzero = s.row_data_.size();

    s.info_.labels = copy_slice(nfo.labels, offset, rows);
    s.info_.root_index = copy_slice(nfo.root_index, offset, rows);
    s.info_.group_ptr = copy_slice(nfo.group_ptr, offset, rows);
    s.info_.weights = copy_slice(nfo.weights, offset, rows);
    s.info_.base_margin = copy_slice(nfo.base_margin, offset, rows);

    offset += rows;
  }

  return slices;
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
}

SliceableMatrix::SliceableMatrix(std::shared_ptr<std::vector<Slice>> slices,
                                 std::vector<size_t> active)
    : slices_(std::move(slices)), active_(std::move(active)) {
  verify(std::all_of(begin(active_), end(active_),
                     [this](size_t a) { return a < slices_->size(); }),
         "invalid active slice");

  size_t offset = 0;
  for (auto const& a : active_) {
    auto const& s = slices_->at(a);

    RowBatch batch;
    batch.size = s.row_count();
    batch.ind_ptr = dmlc::BeginPtr(s.row_ptr_);
    batch.data_ptr = dmlc::BeginPtr(s.row_data_);

    batch.base_rowid = offset;
    offset += s.row_count();

    row_batches_.vec_.push_back(batch);

    info_.num_row += s.info_.num_row;
    info_.num_col = s.info_.num_col;
    info_.num_nonzero += s.info_.num_nonzero;
  }

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

dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator() { return nullptr; }

dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator(
    const std::vector<bst_uint>& fset) {
  return nullptr;
}

void SliceableMatrix::InitColAccess(const std::vector<bool>& enabled,
                                    float subsample, size_t max_row_perbatch) {}
bool SliceableMatrix::HaveColAccess() const { return false; }
bool SliceableMatrix::SingleColBlock() const { return false; }
size_t SliceableMatrix::GetColSize(size_t cidx) const { return 0; }
float SliceableMatrix::GetColDensity(size_t cidx) const { return 0.; };
const RowSet& SliceableMatrix::buffered_rowset() const { return foo_; };

}  // namespace data
}  // namespace xgboost
