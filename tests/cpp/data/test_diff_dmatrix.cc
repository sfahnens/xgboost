// Copyright by Contributors
#include "xgboost/data.h"
#include "../helpers.h"

#include "../../../src/data/diff_dmatrix.h"
#include "../../../src/data/simple_csr_source.h"

using namespace xgboost;
using namespace xgboost::data;

TEST(DiffDMatrix, Simple) {
  auto dmat = CreateDMatrix(20, 100, 0.5);

  std::unique_ptr<SimpleCSRSource> source(new SimpleCSRSource());
  source->CopyFrom(dmat->get());
  auto dmat2 = std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));

  DiffDMatrix(**dmat, *dmat2);

  delete dmat; // WZF
}
