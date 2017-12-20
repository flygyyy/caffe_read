#ifndef CAFFE_BIAS_LAYER_HPP_
#define CAFFE_BIAS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//BiasLayer:计算两个输入Blob的和，后面的Blob如何去匹配前者的shape？？然后进行计算
namespace caffe {

/**
 * @brief Computes a sum of two input Blobs, with the shape of the latter Blob 计算两个输入Blob的和，后面的Blob如何去匹配前者的shape？？然后进行计算
 *        "broadcast" to match the shape of the former. Equivalent to tiling
 *        the latter Blob, then computing the elementwise sum.
 *
 * The second input may be omitted, in which case it's learned as a parameter    第二个输入可能会被忽略，作为这一层的学习参数。
 * of the layer. Note: in case bias and scaling are desired, both operations can 注意：在bias和scaling都是想要得到的参数时，操作能够被ScaleLayer配置层bias_term为true进行处理。
 * be handled by `ScaleLayer` configured with `bias_term: true`.
 */
template <typename Dtype>
class BiasLayer : public Layer<Dtype> {
 public:
  explicit BiasLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Bias"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  Blob<Dtype> bias_multiplier_;
  int outer_dim_, bias_dim_, inner_dim_, dim_;
};



}  // namespace caffe

#endif  // CAFFE_BIAS_LAYER_HPP_
