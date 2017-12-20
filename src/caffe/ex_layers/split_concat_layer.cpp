#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/ex_layers/split_concat_layer.hpp"
#include "caffe/util/math_functions.hpp"
//复制bottom blob并在channel维度合并blob
namespace caffe {

template <typename Dtype>
void SplitConcatLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num_out = this->layer_param_.convolution_param().num_output();//输出 32
  LayerParameter split_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_param));
  split_top_vec_.clear();
  split_top_vec_shared_.resize(num_out);
  for (int index = 0; index < num_out; index++) {
    split_top_vec_shared_[index].reset(new Blob<Dtype>());
    split_top_vec_.push_back(split_top_vec_shared_[index].get());
  }
  split_layer_->SetUp(bottom, split_top_vec_);
  LOG(INFO) << "SplitLayer, top has " << split_top_vec_.size() << " blobs";

  LayerParameter concat_param = this->layer_param();
  concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
  concat_layer_->SetUp(split_top_vec_, top);

}
/*
变形
*/
template <typename Dtype>
void SplitConcatLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Reshape(bottom, split_top_vec_);
  concat_layer_->Reshape(split_top_vec_, top);
}

// 前向传播
// 实现scale矩阵在channel维度上的扩展
template <typename Dtype>
void SplitConcatLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);//复制bottom blob得到
  concat_layer_->Forward(split_top_vec_, top); //将多个bottom blob在channel维度合并
}
// 反向传播
template <typename Dtype>
void SplitConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) { //判断是否需要反向传播
    const int num_out = this->layer_param_.convolution_param().num_output(); //num_output: 32
    const vector<bool> concat_propagate_down(num_out, true); //propagate_down向量判断哪个通道需要反传
    concat_layer_->Backward(top, concat_propagate_down, split_top_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SplitConcatLayer);
#endif

INSTANTIATE_CLASS(SplitConcatLayer);
REGISTER_LAYER_CLASS(SplitConcat);

}  // namespace caffe
