#include <vector>
#include <iostream>
#include "caffe/layers/pattern_conv_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void PatternConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  mod_ = this->layer_param_.convolution_param().mod();
  policy_ = this->layer_param_.convolution_param().policy();
  begin_sp_ = this->layer_param_.convolution_param().begin_sp();
  end_sp_ = this->layer_param_.convolution_param().end_sp();
  iter_pulse_ = this->layer_param_.convolution_param().iter_pulse();
  policy_ = 0;
  current_sp_ = 0;
  int count = this->blobs_[0]->count();
  vector<int> mask_shape(1, count);
  this->masks_.Reshape(mask_shape);
  caffe_set(count, 1, this->masks_.mutable_cpu_data());
  LOG(INFO) << "mod = " << mod_ << std::endl
            << "policy = " << policy_ << std::endl
            << "begin_sp = " << begin_sp_ << std::endl
            << "end_sp = " << end_sp_ << std::endl
            << "iter_pulse = " << iter_pulse_ << std::endl;
}

template <typename Dtype>
void PatternConvLayer<Dtype>::PreProcess() {

}

template <typename Dtype>
void PatternConvLayer<Dtype>::CyclicPrune() {
  if (current_sp_ < end_sp_) {
    if (policy_ == 0) {
      if (iter_ % iter_pulse_ == 0) {
        float last_sp = current_sp_;
        if (current_sp_ == 0) {
          current_sp_ = begin_sp_;
        }
        else if (end_sp_ - current_sp_ <= 5) {
          current_sp_ += 1;
        }
        else if (end_sp_ - current_sp_ <= 15) {
          current_sp_ += 2;
        }
        else if (end_sp_ - current_sp_ <= 30) {
          current_sp_ += 5;
        }
        else {
          current_sp_ += 10;
        }
        LOG(INFO) << "PatternConv Pruning(cyclic), sparsity from "
                  << last_sp << "% to " << current_sp_ << "%" << std::endl;
        // vector<Dtype> weight_row_bag;
        int row_num = this->blobs_[0]->count() / this->blobs_[0]->count(1);
        int row_size = this->blobs_[0]->count(1);
        Dtype* weight_row_bag = new Dtype[row_size / mod_ + 1];
        int global_cnt = 0;
        for (int i = 0; i < row_num; ++i) {
          Dtype* weight_row = this->blobs_[0]->mutable_cpu_data() + i * row_size;
          int* mask_weight_row = this->masks_.mutable_cpu_data() + i * row_size;
          for (int j = 0; j < mod_; ++j) {
            int cnt = 0;
            for (int col = j; col < row_size; col += mod_, ++cnt) {
              weight_row_bag[cnt] = fabs(weight_row[col]);
            }
            std::sort(weight_row_bag, weight_row_bag + cnt);
            Dtype threshold = weight_row_bag[(int)(cnt * (float)(current_sp_ / 100))];
            // set mask_val to 0 to mask it out
            for (int col = j; col < row_size; col += mod_) {
              mask_weight_row[col] = (fabs(weight_row[col]) <= threshold) ? 0 : 1;
              weight_row[col] = (fabs(weight_row[col]) <= threshold) ? 0. : weight_row[col];
              global_cnt += (fabs(weight_row[col]) <= threshold);
            }
            LOG(INFO) << "row " << i << " mod " << j
                      << " threshold: " << threshold <<std::endl;
          }
        }
        float actual_sp = (float)global_cnt / this->blobs_[0]->count();
        LOG(INFO) << "required sparsity: " << current_sp_ << "%, actual sparsity: "
                  << global_cnt << " / " << this->blobs_[0]->count() << " = "
                  << actual_sp << "%" << std::endl;
      }
    }
  }
}


template <typename Dtype>
void PatternConvLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void PatternConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void PatternConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // mask out the gradients
        int count = this->blobs_[0]->count();
        for (int j = 0; j < count; ++j) {
          weight_diff[j] = this->masks_.cpu_data()[j] == 0 ? 0. : weight_diff[j];
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        this->CyclicPrune();
      }
    }
  }
  ++iter_;
}

#ifdef CPU_ONLY
STUB_GPU(PatternConvLayer);
#endif

INSTANTIATE_CLASS(PatternConvLayer);

}  // namespace caffe
