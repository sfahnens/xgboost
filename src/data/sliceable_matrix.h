#pragma once

#include <algorithm>
#include <bitset>
#include <cstdio>
#include <stdexcept>

#include "dmlc/data.h"

#include "../common/group_data.h"
#include "xgboost/data.h"

#ifndef log_err
#define log_err(M, ...) fprintf(stderr, "[ERR] " M "\n", ##__VA_ARGS__);
#endif

#ifdef verify
#undef verify
#endif

#define verify(A, M, ...)        \
  if (!(A)) {                    \
    log_err(M, ##__VA_ARGS__);   \
    throw std::runtime_error(M); \
  }

namespace xgboost {
namespace data {

constexpr auto const kMaxRowsPerBatch = static_cast<size_t>(32UL << 10UL);

struct Page {
  Page() { ptr_.push_back(0); }
  Page(Page const&) = delete;
  Page(Page&&) = default;
  Page& operator=(Page const&) = delete;
  Page& operator=(Page&&) = default;

  SparseBatch::Inst get_inst(size_t idx) {
    return {dmlc::BeginPtr(data_) + ptr_[idx],
            static_cast<bst_uint>(ptr_[idx + 1] - ptr_[idx])};
  }

  SparseBatch::Inst get_inst(size_t idx) const {
    return {dmlc::BeginPtr(data_) + ptr_[idx],
            static_cast<bst_uint>(ptr_[idx + 1] - ptr_[idx])};
  }

  std::vector<size_t> ptr_;
  std::vector<SparseBatch::Entry> data_;
};

using SliceConfigState = std::bitset<64>;

struct Slice {
  Slice() = default;
  Slice(Page rows) : rows_(std::move(rows)) {}
  Slice(Slice const&) = delete;
  Slice(Slice&&) = default;
  Slice& operator=(Slice const&) = delete;
  Slice& operator=(Slice&&) = default;

  size_t row_count() const { return rows_.ptr_.size() - 1; }

  SliceConfigState config_state_;

  // info
  MetaInfo info_;

  // csr data
  Page rows_;

  // csc data
  std::vector<Page> cols_;

  std::vector<size_t> col_offsets_;  // absolute column offset
  std::vector<size_t> col_sizes_;    // number of rows in col page
};

std::vector<Slice> make_slices(DMatrix* mat,
                               std::vector<std::vector<size_t>> const& indices);

template <typename DType>
struct VectorDataIter : public dmlc::DataIter<DType> {
  virtual ~VectorDataIter() = default;

  /*! \brief set before first of the item */
  void BeforeFirst(void) { pos_ = std::numeric_limits<size_t>::max(); }

  /*! \brief move to next item */
  bool Next(void) {
    ++pos_;
    return pos_ < vec_.size();
  }

  /*! \brief get current data */
  const DType& Value(void) const { return vec_.at(pos_); }

  size_t pos_ = std::numeric_limits<size_t>::max();
  std::vector<DType> vec_;
};

struct ColBatchIter : dmlc::DataIter<ColBatch> {
  ColBatchIter()
      : slice_idx_(0), page_idx_(std::numeric_limits<size_t>::max()) {}

  void BeforeFirst() override {
    slice_idx_ = 0;
    page_idx_ = std::numeric_limits<size_t>::max();
  }

  const ColBatch& Value() const override { return batch_; }

  bool Next() override;

  std::vector<Slice*> slices_;
  size_t slice_idx_, page_idx_;

  // data content
  std::vector<bst_uint> col_index_;
  // column content
  std::vector<ColBatch::Inst> col_data_;

  // temporal space for batch
  ColBatch batch_;
};

using slices_vec_ptr = std::shared_ptr<std::vector<Slice>>;

struct SliceableMatrix : public DMatrix {
  SliceableMatrix(slices_vec_ptr slices, std::vector<size_t> active);

  virtual ~SliceableMatrix() = default;
  SliceableMatrix(SliceableMatrix const&) = delete;
  SliceableMatrix(SliceableMatrix&&) = delete;
  SliceableMatrix& operator=(SliceableMatrix const&) = delete;
  SliceableMatrix& operator=(SliceableMatrix&&) = delete;

  /*! \brief meta information of the dataset */
  virtual MetaInfo& info() override;
  /*! \brief meta information of the dataset */
  virtual const MetaInfo& info() const override;
  /*!
   * \brief get the row iterator, reset to beginning position
   * \note Only either RowIterator or  column Iterator can be active.
   */
  virtual dmlc::DataIter<RowBatch>* RowIterator() override;
  /*!\brief get column iterator, reset to the beginning position */
  virtual dmlc::DataIter<ColBatch>* ColIterator() override;
  /*!
   * \brief get the column iterator associated with subset of column features.
   * \param fset is the list of column index set that must be contained in the
   * returning Column iterator
   * \return the column iterator, initialized so that it reads the elements in
   * fset
   */
  virtual dmlc::DataIter<ColBatch>* ColIterator(
      const std::vector<bst_uint>& fset) override;

  void maybe_reindex_column_pages();

  /*!
   * \brief check if column access is supported, if not, initialize column
   * access.
   * \param enabled whether certain feature should be included in column access.
   * \param subsample subsample ratio when generating column access.
   * \param max_row_perbatch auxiliary information, maximum row used in each
   * column batch.
   *         this is a hint information that can be ignored by the
   * implementation.
   * \return Number of column blocks in the column access.
   */
  virtual void InitColAccess(const std::vector<bool>& enabled, float subsample,
                             size_t max_row_perbatch) override;
  // the following are column meta data, should be able to answer them fast.
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess() const override;
  /*! \return Whether the data columns single column block. */
  virtual bool SingleColBlock() const override;
  /*! \brief get number of non-missing entries in column */
  virtual size_t GetColSize(size_t cidx) const override;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const override;
  /*! \return reference of buffered rowset, in column access */
  virtual const RowSet& buffered_rowset() const override;

  SliceConfigState desired_config_state_;
  bool col_access_initialized_;

  slices_vec_ptr slices_;
  std::vector<size_t> active_;

  MetaInfo info_;

  VectorDataIter<RowBatch> row_batches_;
  ColBatchIter col_batches_;

  std::vector<size_t> col_sizes_;

  RowSet row_set_;
};

}  // namespace data
}  // namespace xgboost
