#include "sliceable_matrix.h"

namespace xgboost {

MetaInfo& SliceableMatrix::info() { return info_; }
MetaInfo const& SliceableMatrix::info() const { return info_; }

dmlc::DataIter<RowBatch>* SliceableMatrix::RowIterator() {
  iter_->BeforeForst();
  return iter_.get();
}

dmlc::DataIter<ColBatch>* SliceableMatrix::ColIterator() {

}



} // namespace xgboost
