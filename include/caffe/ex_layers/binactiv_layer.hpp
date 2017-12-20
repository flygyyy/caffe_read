#ifndef CAFFE_BINACTIV_LAYER_HPP_
#define CAFFE_BINACTIV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
//二值激活函数
//对输入进行二值化处理 sign(I) 每个卷积核在输入平移，每个平移区得到一个scale；对整个输入I，得到一个scale矩阵 
namespace caffe {
//Parameter：propagate_down是否需要回传，如果需要固定参数的话，propagate_down设为0
/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results. 从num或者channel维度分割Blob
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BinActivLayer : public Layer<Dtype> {
 public:
  explicit BinActivLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinActiv"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal Convolution layer  内部卷积
  // 计算输入 I 的系数矩阵K
  shared_ptr<Layer<Dtype> > convolution_layer_; //卷积层
  vector<Blob<Dtype>*> convolution_bottom_vec_;
  shared_ptr<Blob<Dtype> >  convolution_bottom_shared_;
  vector<Blob<Dtype>*> convolution_top_vec_;
  shared_ptr<Blob<Dtype> >  convolution_top_shared_;
};

}  // namespace caffe

#endif  // CAFFE_BINACTIV_LAYER_HPP_
