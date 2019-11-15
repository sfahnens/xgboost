#include "reconfigurable_matrix.h"

#include "../common/group_data.h"

#include "vec_helper.h"

namespace xgboost {
namespace data {

template <bool UseRowPage>
struct ReconfigurableBatchIteratorImpl : public BatchIteratorImpl {
  explicit ReconfigurableBatchIteratorImpl(
      ReconfigurableSource* src, ReconfigurableBatch::ConfigState config_state)
      : src_{src},
        config_state_{config_state},
        lim_{std::min(src_->batches_.size(), config_state.size())} {
    CHECK(src_ != nullptr);
    Advance();
  }
  SparsePage& operator*() override {
    auto& batch = src_->batches_[idx_];
    CHECK(batch.config_state_ == config_state_) << "invalid config_state";
    return UseRowPage ? *batch.rows_ : *batch.cols_;
  }
  const SparsePage& operator*() const override {
    auto const& batch = src_->batches_[idx_];
    CHECK(batch.config_state_ == config_state_) << "invalid config_state";
    return UseRowPage ? *batch.rows_ : *batch.cols_;
  }
  void operator++() override {
    ++idx_;
    Advance();
  }
  bool AtEnd() const override { return idx_ >= lim_; }
  ReconfigurableBatchIteratorImpl<UseRowPage>* Clone() override {
    return new ReconfigurableBatchIteratorImpl<UseRowPage>(*this);
  }

  void Advance() {
    for (; idx_ < lim_; ++idx_) {
      if (config_state_.test(idx_)) {
        break;
      }
    }
  }

  ReconfigurableSource* src_{nullptr};
  ReconfigurableBatch::ConfigState config_state_;
  size_t idx_{0};
  size_t lim_;
};

template <typename Field>
void MergeVector(MetaInfo&, std::vector<ReconfigurableBatch> const&,
                 std::vector<size_t> const&, Field);

ReconfigurableMatrix::ReconfigurableMatrix(ReconfigurableSourcePtr source,
                                           std::vector<size_t> active)
    : source_{std::move(source)} {
  CHECK(std::all_of(begin(active), end(active), [this](size_t const a) {
    return a < source_->batches_.size();
  })) << "invalid active batch";

  info_.Clear();

  // ensure that info_ is synced with this_config_state_
  std::sort(begin(active), end(active));

  CHECK(!active.empty()) << "need at least one active batch!";
  info_.num_col_ = source_->batches_.at(active.front()).info_.num_col_;
  col_sizes_.resize(info_.num_col_, 0ul);

  for (auto const& a : active) {
    this_config_state_.set(a);

    auto const& s = source_->batches_.at(a);
    info_.num_row_ += s.info_.num_row_;
    info_.num_nonzero_ += s.info_.num_nonzero_;

    CHECK(info_.num_col_ == s.info_.num_col_) << "col count mismatch";

    for(auto const& e : s.rows_->data.HostVector()) {
      ++col_sizes_[e.index];
    }
  }

  MergeVector(info_, source_->batches_, active, &MetaInfo::labels_);
  MergeVector(info_, source_->batches_, active, &MetaInfo::root_index_);
  MergeVector(info_, source_->batches_, active, &MetaInfo::group_ptr_);
  MergeVector(info_, source_->batches_, active, &MetaInfo::weights_);
  MergeVector(info_, source_->batches_, active, &MetaInfo::base_margin_);
}

template <typename Field>
void MergeVector(MetaInfo& info,
                 std::vector<ReconfigurableBatch> const& batches,
                 std::vector<size_t> const& active, Field field) {
  if (vec(batches.at(active.at(0)).info_.*field).empty()) {
    return;
  }

  vec(info.*field).resize(info.num_row_);  // XXX ?
  size_t offset = 0;
  for (auto const& a : active) {
    auto const& b = batches.at(a);

    std::copy(std::begin(vec(b.info_.*field)), std::end(vec(b.info_.*field)),
              std::begin(vec(info.*field)) + offset);
    offset += b.info_.num_row_;
  }
  CHECK(vec(info.*field).size() == offset) << "size mismatch";
}

MetaInfo& ReconfigurableMatrix::Info() { return info_; }
MetaInfo const& ReconfigurableMatrix::Info() const { return info_; }

BatchSet ReconfigurableMatrix::GetRowBatches() {
  source_->MaybeReconfigure(this_config_state_);
  return BatchSet{BatchIterator{new ReconfigurableBatchIteratorImpl<true>(
      source_.get(), this_config_state_)}};
}

BatchSet ReconfigurableMatrix::GetColumnBatches() {
  throw std::runtime_error{"not implemented"};
}

BatchSet ReconfigurableMatrix::GetSortedColumnBatches() {
  source_->LazyInitializeColumns();
  source_->MaybeReconfigure(this_config_state_);
  return BatchSet{BatchIterator{new ReconfigurableBatchIteratorImpl<false>(
      source_.get(), this_config_state_)}};
}

bool ReconfigurableMatrix::SingleColBlock() const {
  return source_->batches_.size() == 1;
}

float ReconfigurableMatrix::GetColDensity(size_t cidx) {
  size_t nmiss = info_.num_row_ - col_sizes_[cidx];
  return 1.0f - (static_cast<float>(nmiss)) / info_.num_row_;
}

}  // namespace data
}  // namespace xgboost
