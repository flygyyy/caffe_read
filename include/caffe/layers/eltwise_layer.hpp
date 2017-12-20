#ifndef CAFFE_ELTWISE_LAYER_HPP_
#define CAFFE_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//计算elementwise操作，例如多个输入的点、和
namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
 public:
  explicit EltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Eltwise"; }
  virtual inline int MinBottomBlobs() const { return 2; } //最少输入Blob为2
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  EltwiseParameter_EltwiseOp op_; //操作类型
  vector<Dtype> coeffs_; //SUM操作的常量因子
  Blob<int> max_idx_; //the vector index part

  bool stable_prod_grad_;  //乘法操作时是否采用渐进方式计算梯度
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
