#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
//Data_layer的实现基础
namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 * Data_layer的实现基础
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial. 数据层不包含bottoms，变形较简单
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
 //反向传播不需任何操作，没有正向传播？？？
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  //数据预处理变换器参数
  TransformationParameter transform_param_;
  //数据预处理变换器
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  //是否输出标签数据
  bool output_labels_;
};
//批量数据，用于传输数据读取层输出
template <typename Dtype>
class Batch {
 public:
  //包含两个blob，一个用于存放数据，一个用于存放标签
  Blob<Dtype> data_, label_;
};
//带预取功能的数据读取层，派生于 BaseDataLayer, public InternalThread
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  //内部线程入口
  virtual void InternalThreadEntry();
  //载入批量数据，纯虚函数
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  vector<shared_ptr<Batch<Dtype> > > prefetch_;
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;//变换后的数据
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
