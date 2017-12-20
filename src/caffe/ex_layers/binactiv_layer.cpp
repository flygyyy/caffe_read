#include <algorithm>
#include <vector>

#include "caffe/ex_layers/binactiv_layer.hpp"
#include "caffe/util/math_functions.hpp"
// 论文中XNOR卷积层的结构是BN+BinActiv+BinConv+Pooling
// XNOR Net的激活层
// 输出2个top blob  对输入进行二值化处理
namespace caffe {

#define sign(x) ((x)>=0?1:-1)  //大于等于0取1，小于0取-1

template <typename Dtype>
void BinActivLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 2); //检查是否2个top Blob
  CHECK_EQ(bottom.size(), 1);//检查是否1个bottom Blob
  LayerParameter convolution_param(this->layer_param_); //获取卷积层参数 kernel:5*5
  convolution_param.set_type("Convolution");//卷积层
  convolution_param.mutable_convolution_param()->CopyFrom( //从prototxt参数复制卷积参数
    this->layer_param_.convolution_param() );
  convolution_layer_ = LayerRegistry<Dtype>::CreateLayer(convolution_param);//根据卷积参数convolution_param创建卷积层
  // 对卷积层的bottom blob和top blob进行重置
  convolution_bottom_shared_.reset(new Blob<Dtype>());
  convolution_bottom_vec_.clear();
  convolution_bottom_vec_.push_back(convolution_bottom_shared_.get());
  convolution_bottom_vec_[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width()); //1 channel
  convolution_top_shared_.reset(new Blob<Dtype>());
  convolution_top_vec_.clear();
  convolution_top_vec_.push_back(convolution_top_shared_.get());
  convolution_layer_->SetUp(convolution_bottom_vec_, convolution_top_vec_); //根据bottom blob和top blob设置卷积层
}
/*
变形
*/ 
template <typename Dtype>
void BinActivLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ",,, enter Reshape";
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(bottom.size(), 1);
  //top[0]的size与输入size相同，top[1]的size，在channel维为1，但是height和width应该是经过卷积后的尺度？？？
  top[0]->ReshapeLike(*bottom[0]);
  convolution_bottom_vec_[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  top[1]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width()); 
  convolution_layer_->Reshape(convolution_bottom_vec_, convolution_top_vec_); //计算卷积层的bottom blob和top blob的size 
  top[1]->ReshapeLike(*convolution_top_vec_[0]);
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", bottom[0] shape : " << bottom[0]->num() << ", " << bottom[0]->channels() << ", " << bottom[0]->height() << ", " << bottom[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", top[0] shape : " << top[0]->num() << ", " << top[0]->channels() << ", " << top[0]->height() << ", " << top[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", top[1] shape : " << top[1]->num() << ", " << top[1]->channels() << ", " << top[1]->height() << ", " << top[1]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", conv_bottom_vec[0] shape : " << convolution_bottom_vec_[0]->num() << ", " << convolution_bottom_vec_[0]->channels() << ", " << convolution_bottom_vec_[0]->height() << ", " << convolution_bottom_vec_[0]->width();
  DLOG(INFO) << "-----> " << this->layer_param_.name() << ", conv_top_vec[0] shape : " << convolution_top_vec_[0]->num() << ", " << convolution_top_vec_[0]->channels() << ", " << convolution_top_vec_[0]->height() << ", " << convolution_top_vec_[0]->width();
}
//forward计算
//修改本段代码使得top[1]的size和卷积输出top的代码一致
template <typename Dtype>
void BinActivLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  //top[1]的size和convolution_top_vec_[0]的size应该一致  BUG需要完善
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* sumA = convolution_bottom_vec_[0]->mutable_cpu_data();
  Dtype* signA = top[0]->mutable_cpu_data();
  const int C = bottom[0]->channels();
  for (int index = 0; index < bottom[0]->num(); index++ ) {
    for (int _h = 0; _h < bottom[0]->height(); _h++ ) {
      for (int _w = 0; _w < bottom[0]->width(); _w++ ) { //convolution_bottom_vec_的shape为 num_*1*height_*width_
        sumA[ convolution_bottom_vec_[0]->offset(index,0,_h,_w) ] = 0; //sumA数组置0，sumA的channel为1
        for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
          sumA[ convolution_bottom_vec_[0]->offset(index,0,_h,_w) ] +=  //求channel上的均值
            bottom_data[ bottom[0]->offset(index,_c,_h,_w) ] / C;
          signA[ top[0]->offset(index,_c,_h,_w)] = //signA为输入的符号，进行二值化处理
            sign(bottom_data[ bottom[0]->offset(index,_c,_h,_w) ]);
        }
      }
    }
  }
  convolution_layer_->Forward(convolution_bottom_vec_, convolution_top_vec_);  //sumA为bottom blob，权重为常数 weight_filler初始化为1， 并且merge Layer 设置K-norm1的  propagate_down: 0
  const int size_kernal = this->layer_param_.convolution_param().kernel_size(0)
        * this->layer_param_.convolution_param().kernel_size(0); //kernel有多少个元素
  //top[1]->ReshapeLike(*convolution_top_vec_[0]);    
  CHECK_EQ(top[1]->count(), convolution_top_vec_[0]->count());//检查top[1]和convolution_top_vec_[0]的size是否一致
  caffe_copy(top[1]->count(), convolution_top_vec_[0]->cpu_data(), top[1]->mutable_cpu_data());
  caffe_cpu_scale(top[1]->count(), Dtype(1)/size_kernal, top[1]->cpu_data(),  top[1]->mutable_cpu_data()); //scale除以wxh
}

/*
反向传播
BinactivLayer共由两个top blob，一个输出二值卷积，另一个为Input的scale factor矩阵
在Eltwise层，设置K-norm的propagate_down: 0，不会向前传播
只更新B-norm
*/
template <typename Dtype>
void BinActivLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int index = 0; index < bottom[0]->count(); index++) {
    if ( std::abs(bottom_data[index]) <= Dtype(1) ) {
      bottom_diff[ index ] = top_diff[ index ];
    } else {
      bottom_diff[ index ] = Dtype(0);//？？？与论文中的表述是一样的，如果按照实际的sign(I)计算梯度，大部分都是0，无法反向传播
    }
  } 
}

#ifdef CPU_ONLY
STUB_GPU(BinActivLayer);
#endif

INSTANTIATE_CLASS(BinActivLayer);
REGISTER_LAYER_CLASS(BinActiv);

}  // namespace caffe
