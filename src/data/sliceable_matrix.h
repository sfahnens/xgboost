#pragma once

namespace xgboost {


struct SliceableMatrix : public DMatrix {

  /*! \brief meta information of the dataset */
  virtual MetaInfo& info();
  /*! \brief meta information of the dataset */
  virtual const MetaInfo& info() const;
  /*!
   * \brief get the row iterator, reset to beginning position
   * \note Only either RowIterator or  column Iterator can be active.
   */
  virtual dmlc::DataIter<RowBatch>* RowIterator();
  /*!\brief get column iterator, reset to the beginning position */
  virtual dmlc::DataIter<ColBatch>* ColIterator();
  /*!
   * \brief get the column iterator associated with subset of column features.
   * \param fset is the list of column index set that must be contained in the returning Column iterator
   * \return the column iterator, initialized so that it reads the elements in fset
   */
  virtual dmlc::DataIter<ColBatch>* ColIterator(const std::vector<bst_uint>& fset);
  /*!
   * \brief check if column access is supported, if not, initialize column access.
   * \param enabled whether certain feature should be included in column access.
   * \param subsample subsample ratio when generating column access.
   * \param max_row_perbatch auxiliary information, maximum row used in each column batch.
   *         this is a hint information that can be ignored by the implementation.
   * \return Number of column blocks in the column access.
   */
  virtual void InitColAccess(const std::vector<bool>& enabled,
                             float subsample,
                             size_t max_row_perbatch);
  // the following are column meta data, should be able to answer them fast.
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess() const;
  /*! \return Whether the data columns single column block. */
  virtual bool SingleColBlock() const;
  /*! \brief get number of non-missing entries in column */
  virtual size_t GetColSize(size_t cidx) const;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const;
  /*! \return reference of buffered rowset, in column access */
  virtual const RowSet& buffered_rowset() const;


  std::unique_ptr<MetaInfo> info_;

  std::vector<size_t> row_ptr_;
  std::vector<RowBatch::Entry> row_data_;

  std::unique_ptr<dmlc::DataIter<RowBatch>> iter_;

  std::vector<size_t> col_ptr_;
  std::vector<ColBatch::Entry> col_data_;

};


} // namespace xgboost
