#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//损失层
namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
 //两个blobs作为输入，一个预测一个真实标签
template <typename Dtype>
class LossLayer : public Layer<Dtype> {//派生于Layer
 public:
  explicit LossLayer(const LayerParameter& param)//显式构造函数
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);//层配置函数
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);//变形函数

  virtual inline int ExactNumBottomBlobs() const { return 2; }//接受两个Blob作为输入

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
   //为了方便和向后兼容，指导网络自动为损失层分配一个输出Blob，损失层将计算结果保存在此
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }//只有一个输出Blob
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
   //我们经常不能对标签做反向计算，忽略force_backward
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
