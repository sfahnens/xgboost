#include "sliceable_matrix.h"

namespace xgboost {

// SliceableMatrix make_sliceable_matrix(DMatrix const& src) {

//   return {src->info(), };
// }

std::vector<Slice> matrix_to_slices(DMatrix* mat, size_t nrow) {
  std::vector<Slice> slices;
  slices.emplace_back();

  dmlc::DataIter<RowBatch>* it = mat->RowIterator();
  it->BeforeFirst();
  while(it->Next()) {
    auto const& batch = it->Value();
    for (size_t i = 0; i < batch.size; ++i) {
      auto& s = slices.back();
      RowBatch::Inst inst = batch[i];
      s.row_data_.insert(s.row_data_.end(), inst.data, inst.data + inst.length);
      s.row_ptr_.push_back(s.row_ptr_.back() + inst.length);

      if(s.row_ptr_.size() - 1 == nrow) {
        slices.emplace_back();
      }
    }
  }

  if(slices.back().row_ptr_.size() == 1) {
    slices.erase(end(slices)-1);
  }
  // verify slices not empty

  size_t offset = 0;
  for(auto& s : slices) {
    // verify slice not empty

    s.batch_.size = s.row_ptr_.size() - 1;
    s.batch_.ind_ptr = dmlc::BeginPtr(s.row_ptr_);
    s.batch_.data_ptr = dmlc::BeginPtr(s.row_data_);
    
    s.batch_.base_rowid = offset;
    offset += s.batch_.size;

    // TODO copy meta info

  }

  return slices;
}



// MetaInfo& SliceableMatrix::info() { return info_; }
// MetaInfo const& SliceableMatrix::info() const { return info_; }

// dmlc::DataIter<RowBatch>* SliceableMatrix::RowIterator() {
//   iter_->BeforeFirst();
//   return iter_.get();
// }

// dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator() {

// }



} // namespace xgboost
