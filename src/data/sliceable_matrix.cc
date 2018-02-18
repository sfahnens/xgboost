#include "sliceable_matrix.h"

namespace xgboost {
namespace data {

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
  // verify slices not empty

  // size_t offset = 0;
  // for(auto& s : slices) {
  //   // verify slice not empty

  //   s.batch_.size = s.row_ptr_.size() - 1;
  //   s.batch_.ind_ptr = dmlc::BeginPtr(s.row_ptr_);
  //   s.batch_.data_ptr = dmlc::BeginPtr(s.row_data_);

  //   s.batch_.base_rowid = offset;
  //   offset += s.batch_.size;

  //   // TODO copy meta info

  // }

  return slices;
}

SliceableMatrix::SliceableMatrix(std::shared_ptr<std::vector<Slice>> slices,
                                 std::vector<size_t> active)
    : slices_(std::move(slices)), active_(std::move(active)) {
  verify(std::all_of(begin(active_), end(active_),
                     [this](size_t a) { return a < slices_->size(); }),
         "invalid active slice");

  size_t offset = 0;
  for (auto const& a : active_) {
    auto& s = slices_->at(a);

    RowBatch batch;
    batch.size = s.row_ptr_.size() - 1;
    batch.ind_ptr = dmlc::BeginPtr(s.row_ptr_);
    batch.data_ptr = dmlc::BeginPtr(s.row_data_);

    batch.base_rowid = offset;
    offset += batch.size;

    row_batches_.vec_.push_back(batch);
  }
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
