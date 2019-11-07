#pragma once

#include "xgboost/data.h"

namespace xgboost {
namespace data {

void DiffMeta(MetaInfo const& a, MetaInfo const& b);

void DiffDMatrixByRowNotEmpty(DMatrix& a, DMatrix& b);

void DiffDMatrix(DMatrix& a, DMatrix& b);

}  // namespace data
}  // namespace xgboost
