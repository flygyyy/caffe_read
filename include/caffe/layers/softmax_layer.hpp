#ifndef CAFFE_SOFTMAX_LAYER_HPP_
#define CAFFE_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
// Softmax层，计算Softmax函数 
namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)//显式构造函数
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,//变形函数
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmax"; }//返回层的名字
  virtual inline int ExactNumBottomBlobs() const { return 1; }//获取bottom和top blob的数量状态
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  //前向传播和后向传播
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS 利用BLAS计算求和
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results. 用于存储计算中间结果的Blob,scale
  Blob<Dtype> scale_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
