#pragma once

#include "xgboost/data.h"

#include "reconfigurable_source.h"

namespace xgboost {
namespace data {

struct ReconfigurableMatrix : public DMatrix {
  ReconfigurableMatrix(ReconfigurableSourcePtr source,
                       std::vector<size_t> active);
  virtual ~ReconfigurableMatrix() = default;

  MetaInfo& Info() override;
  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;
  bool SingleColBlock() const override;

  BatchSet GetRowBatches() override;
  BatchSet GetColumnBatches() override;
  BatchSet GetSortedColumnBatches() override;

  ReconfigurableSourcePtr source_;
  ReconfigurableBatch::ConfigState this_config_state_{0};

  MetaInfo info_;
  std::vector<size_t> col_sizes_;
};

}  // namespace data
}  // namespace xgboost
