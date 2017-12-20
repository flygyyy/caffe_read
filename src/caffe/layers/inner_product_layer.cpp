#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
//全连接层
namespace caffe {  
/*
全连接层初始化
输入层：（M_, K_, 1, 1); 
输出层： (M_, N_, 1, 1); 
W矩阵：（N_,K_,1,1); 
b矩阵：（N_,1,1,1); 
M_样本个数，K_单个样本特征长度，N_全连接之后神经元的个数。
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output(); //输出channel的数目
  bias_term_ = this->layer_param_.inner_product_param().bias_term();//获取偏置信息
  transpose_ = this->layer_param_.inner_product_param().transpose();//转置信息
  N_ = num_output;//输出channel的数目
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());//标准化索引，主要是对参数索引进行标准化，以满足要求  channel轴对应的axis number
  // 从axis轴开始平铺层一个大小为K_的向量
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis); //axis=1，channel轴的大小，有多少个channel
  // 检查是否需要设置权重
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) { //size不为0说明已经初始化
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) { //是否包含偏置项
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // 初始化权重
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {//是否转置
      weight_shape[0] = K_; //输入
      weight_shape[1] = N_; //输出
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // 填充权重
    // fill the weights  
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // 存在偏置，进行初始化
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}
/*
变形
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis()); //axis = 1
  const int new_K = bottom[0]->count(axis); //channel大小 count是否 axis X width X height
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";//输入size与全连接层的参数不同
  // 第一个axis与全连接层独立， 大小为M_
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis); //计算从axis = 0 到 1的 数目(不包含1???)

  // top blob的shape 在channel用 N_替换，其它与bottom一致
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1); //
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // bias进行scale的系数，通常为1，即表示不进行scale
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_); //？？？为什么是M_, M_为输入batch size
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data()); //用常数 alpha 对 Y 进行初始化 
  }
}
/*
前向推理
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); //输入数据， 常量
  Dtype* top_data = top[0]->mutable_cpu_data(); //输出数据， 指针
  const Dtype* weight = this->blobs_[0]->cpu_data();//权重数据
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans, //根据bottom_data、weight 得到输出top_data
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data); 
  if (bias_term_) {  //存在偏置加上偏置
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}
/*
反向传播
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) { //权重是否需要更新
    const Dtype* top_diff = top[0]->cpu_diff(); //top blob的梯度
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) { //是否转置
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff()); //根据top_diff、bottom_data计算blobs_[0]的梯度
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) { //bias是否需要更新
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff, //计算bias的梯度
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) { //是否向前面的Layer传播
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {//是否转置
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,   //根据top 梯度、 权重计算bottom梯度
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(), 
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
