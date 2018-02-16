// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/sliceable_matrix.h"

#include "../helpers.h"

TEST(SliceableMatrix, MatrixToSlices) {
  std::string tmp_file = CreateSimpleTestData();
  auto const dmat = std::unique_ptr<xgboost::DMatrix>{xgboost::DMatrix::Load(tmp_file, true, false)};
  std::remove(tmp_file.c_str());

  auto const& slices = matrix_to_slices(dmat.get(), 1);
  ASSERT_EQ(slices.size(), 2);

  // TODO assert content ...

}
