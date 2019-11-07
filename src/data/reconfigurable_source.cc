#include "reconfigurable_source.h"

#include "vec_helper.h"

namespace xgboost {
namespace data {

void FinalizeBatches(std::vector<ReconfigurableBatch>&, MetaInfo const&,
                     std::vector<std::vector<size_t>> const&);
template <typename Vec>
void CopyInfoBatch(MetaInfo const&, MetaInfo&, std::vector<size_t> const&, Vec);

ReconfigurableSourcePtr ReconfigurableSource::Create(
    DMatrix* mat, std::vector<std::vector<size_t>> const& indices) {
  CHECK(indices.size() <= ReconfigurableBatch::ConfigState{}.size())
      << "too many batches requested";

  std::vector<ReconfigurableBatch> batches;
  for (auto const& idx : indices) {
    size_t curr = 0;
    size_t idx_pos = 0;

    SparsePage dest_page;
    for (auto const& src_page : mat->GetRowBatches()) {
      for (size_t i = 0; i < src_page.Size(); ++i) {
        if (idx_pos >= idx.size()) {
          break;
        }
        if (curr++ < idx[idx_pos]) {
          continue;
        }
        ++idx_pos;

        dest_page.Push(src_page[i]);
      }
    }

    CHECK(dest_page.offset.Size() - 1 == idx.size())
        << "not all rows found! (rows=" << dest_page.offset.Size() - 1
        << ", idx=" << idx.size() << ")";
    batches.emplace_back(std::move(dest_page));
  }

  FinalizeBatches(batches, mat->Info(), indices);
  return std::make_shared<ReconfigurableSource>(std::move(batches));
}

ReconfigurableSourcePtr ReconfigurableSource::Create(
    size_t nrow, std::vector<size_t> const& col_widths,
    std::vector<ColCreatorFn> const& col_creators, double const* labels,
    double const* weights, std::vector<std::vector<size_t>> const& indices) {
  CHECK(indices.size() <= ReconfigurableBatch::ConfigState{}.size())
      << "too many batches requested";

  std::vector<size_t> col_offsets;
  col_offsets.push_back(0);
  for (auto const& w : col_widths) {
    col_offsets.push_back(w + col_offsets.back());
  }

  std::vector<ReconfigurableBatch> batches;
  for (auto const& idx : indices) {
    SparsePage dest_page;
    auto& offset = dest_page.offset.HostVector();
    auto& data = dest_page.data.HostVector();

    // all rows are present - first zero already set
    offset.reserve(idx.size() + 1);

    // good approximation (only numeric 0 and logic FALSE to much)
    data.reserve(col_creators.size() * idx.size());

    for (auto const& row_idx : idx) {
      CHECK(row_idx < nrow) << "invalid index";

      size_t row_size = 0;
      for (auto c = 0ul; c < col_creators.size(); ++c) {
        bst_uint index = col_offsets[c];
        bst_float fvalue = 0.;

        col_creators[c](row_idx, &index, &fvalue);

        if (fvalue != 0.) {
          data.emplace_back(index, fvalue);
          ++row_size;
        }
      }
      offset.push_back(offset.back() + row_size);
    }

    data.shrink_to_fit();
    batches.emplace_back(std::move(dest_page));
  }

  // finalize batches
  MetaInfo nfo;
  nfo.num_row_ = nrow;
  nfo.num_col_ = col_offsets.back();

  nfo.labels_.HostVector().assign(labels, labels + nrow);
  if (weights != nullptr) {
    nfo.weights_.HostVector().assign(weights, weights + nrow);
  }

  FinalizeBatches(batches, nfo, indices);
  return std::make_shared<ReconfigurableSource>(std::move(batches));
}

void FinalizeBatches(std::vector<ReconfigurableBatch>& batches,
                     MetaInfo const& nfo,
                     std::vector<std::vector<size_t>> const& indices) {
  CHECK(batches.size() == indices.size()) << "batch/indices size mismatch";

  size_t offset = 0;
  for (auto i = 0ul; i < batches.size(); ++i) {
    auto& s = batches.at(i);
    s.config_state_.set(i);  // XXX !?

    auto const& idx = indices.at(i);
    CHECK(idx.size() == s.rows_.Size())
        << "row size mismatch idx=" << idx.size() << " data=" << s.rows_.Size();

    s.info_.num_row_ = s.rows_.Size();
    s.info_.num_col_ = nfo.num_col_;
    s.info_.num_nonzero_ = s.rows_.data.Size();

    CopyInfoBatch(nfo, s.info_, idx, &MetaInfo::labels_);
    CopyInfoBatch(nfo, s.info_, idx, &MetaInfo::root_index_);
    CopyInfoBatch(nfo, s.info_, idx, &MetaInfo::group_ptr_);
    CopyInfoBatch(nfo, s.info_, idx, &MetaInfo::weights_);
    CopyInfoBatch(nfo, s.info_, idx, &MetaInfo::base_margin_);

    s.cols_ = s.rows_.GetTranspose(s.info_.num_col_);
    s.cols_.SortRows();

    offset += s.rows_.Size();
  }
  CHECK(offset == nfo.num_row_)
      << "row count mismatch" << offset << " != " << nfo.num_row_;
}

template <typename Vec>
void CopyInfoBatch(MetaInfo const& src, MetaInfo& dst,
                   std::vector<size_t> const& indices, Vec v) {
  if (vec(src.*v).empty()) {
    return;
  }

  vec(dst.*v).reserve(indices.size());
  for (auto const& i : indices) {
    vec(dst.*v).push_back(vec(src.*v).at(i));
  }
}

void ReconfigurableSource::MaybeReconfigure(
    ReconfigurableBatch::ConfigState target_state) {
  size_t offset = 0;
  for (auto i = 0; i < batches_.size(); ++i) {
    auto& s = batches_[i];
    if (!target_state.test(i) || s.config_state_ == target_state) {
      continue;
    }

    auto old_offset = s.rows_.base_rowid;

    // rows: just set global offset
    s.rows_.base_rowid = offset;

    // cols: shift each entry -> subtract old offset; add new offset
    for (auto& entry : s.cols_.data.HostVector()) {
      entry.index += -old_offset + offset;
    }

    offset += s.info_.num_row_;
    s.config_state_ = target_state;
  }
}

bool ReconfigurableSource::Next() { return false; }

void ReconfigurableSource::BeforeFirst() {
  throw std::runtime_error{
      "not implemeted: ReconfigurableSource::BeforeFirst()"};
}
const SparsePage& ReconfigurableSource::Value() const {
  throw std::runtime_error{"not implemeted: ReconfigurableSource::Value()"};
  return batches_.at(0).rows_;
}

}  // namespace data
}  // namespace xgboost
