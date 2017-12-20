#include <vector>

#include "caffe/filler.hpp"
#include "caffe/ex_layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
//二值化全连接层
namespace caffe {  

#define sign(x) ((x)>=0?1:-1)  //大于等于0取1，小于0取-1

/*
全连接层初始化
输入层：（M_, K_, 1, 1); 
输出层： (M_, N_, 1, 1); 
W矩阵：（N_,K_,1,1); 
b矩阵：（N_,1,1,1); 
M_样本个数，K_单个样本特征长度，N_全连接之后神经元的个数。
*/
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  K_ = bottom[0]->count(axis); //axis=1，CHW的向量
  // 检查是否需要设置权重
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) { //size不为0说明已经初始化
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) { //是否包含偏置项， 应该去除偏置项
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // 初始化权重
    // Initialize the weights
    vector<int> weight_shape(2); //shape为2
    if (transpose_) {//是否转置  转置为K_*N_
      weight_shape[0] = K_; //输入
      weight_shape[1] = N_; //输出
    } else {
      weight_shape[0] = N_; //没有转置为N_*K_
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape)); //重置weight的shape
    // 填充权重
    // fill the weights  
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // 存在偏置，进行初始化
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_); //全连接层的bias，shape为1*N_(N*1*1*1)
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // For Binary
  this->W_b = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->W_buffer = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->W_b->ReshapeLike(*(this->blobs_[0]));
  this->W_buffer->ReshapeLike(*(this->blobs_[0]));

  this->I_b = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->I_buffer = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->I_b->ReshapeLike(*(bottom[0]));
  this->I_buffer->ReshapeLike(*(bottom[0]));
}
/*
变形
*/
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis()); //axis = 1
  const int new_K = bottom[0]->count(axis); //count是否 axis X width X height
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";//输入size与全连接层的参数不同
  // 第一个axis与全连接层独立， 大小为M_
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis); //计算从axis = 0 到 1的 数目(不包含1???)

  // top blob的shape 在channel用 N_替换，其它与bottom一致
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape(); //复制bottom blob的shape, bottom shape 大小不一定如果之前为全连接层则为2，否则为4
  top_shape.resize(axis + 1); //top_shape = 2
  top_shape[axis] = N_; //第一个axis为复制bottom的M_,第二个axis为N_
  top[0]->Reshape(top_shape);// 大小为2
  // bias进行scale的系数，通常为1，即表示不进行scale
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_); //？？？为什么是M_, M_为输入batch size  M_*1还是1×M_
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data()); //用常数 alpha 对 Y 进行初始化 
  }
  this->W_b->ReshapeLike(*(this->blobs_[0]));
  this->W_buffer->ReshapeLike(*(this->blobs_[0]));
  this->I_b->ReshapeLike(*(bottom[0]));
  this->I_buffer->ReshapeLike(*(bottom[0]));
}
/*
前向推理
*/
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data(); //输出数据， 指针

  //对输入进行二值化
  //放在hpp头文件中
  std::vector<Dtype> scaleA(bottom[0]->num(), 0); //输出通道的个数num()，存储权重scale  
  int index;
  //自定义函数对输入和权重进行二值化处理，得到sign(I)和sign(W)，并得到scale因子，判断输入数据的格式
  if(bottom[0]->channels() == K_){//shape为2
    for (int _num = 0; index < bottom[0]->num(); _num++ ) {
      for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
        index = bottom[0]->offset(_num,_c);        
        scaleA[ _num ] += std::abs(bottom[0]->cpu_data()[index]) / Dtype(K_);
        I_b->mutable_cpu_data()[index] = sign(bottom[0]->cpu_data()[index]);              
      }
    }
  }
  else { //shape为4
    for (int _num = 0; index < bottom[0]->num(); _num++ ) {
      for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
        for (int _h = 0; _h < bottom[0]->height(); _h++ ) {
          for (int _w = 0; _w < bottom[0]->width(); _w++ ) {
            index = bottom[0]->offset(index,_c,_h,_w);            
            scaleA[ _num ] += std::abs(bottom[0]->cpu_data()[index]) / K_;
            I_b->mutable_cpu_data()[index] = sign(bottom[0]->cpu_data()[index]);              
          }
        }
      }
    }
  }
  //copyCPUTo(bottom[0], I_buffer);//全精度的权重blobs_[0]暂存到W_buffer
  //copyCPUTo(I_b, bottom[0]);//复制二值权重到权重参数

  // 对权重进行二值化 权重的shape就是2
  std::vector<Dtype> scaleW(this->blobs_[0]->num(), 0); //输出通道的个数num()，存储权重scale    
  for (int _num = 0; _num < this->blobs_[0]->num(); _num++) {
    for (int _c = 0; _c < this->blobs_[0]->channels(); _c++){
      index = this->blobs_[0]->offset(_num,_c);
      scaleW[_num] += std::abs(this->blobs_[0]->gpu_data()[index]) / Dtype(K_); //计算L1范数并除以元素个数，得到weight的scale因子， 一共有num_个scale
      W_b->mutable_cpu_data()[index] = sign(this->blobs_[0]->cpu_data()[index]);
    }
  }
  copyCPUTo(this->blobs_[0], W_buffer);//全精度的权重blobs_[0]暂存到W_buffer
  copyCPUTo(W_b, this->blobs_[0]);//复制二值权重到权重参数

  //const Dtype* bottom_data = bottom[0]->cpu_data(); //输入数据， 常量  
  const Dtype* bottom_binary = I_b->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();//读取权重数据
  //输入和权重的二值计算 利用I_b和W_b进行计算
  //根据bottom_data、weight 得到输出top_data M_xK_ * K_*N_   如果transpose本出不转K_*N_；否则转置
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans, 
      M_, N_, K_, (Dtype)1.,
      bottom_binary, weight, (Dtype)0., top_data); 
  if (bias_term_) {  //存在偏置加上偏置   M_*1  1*N_   M_*N_, 加上bias_multiplier 与bias相乘得到M_*N_的矩阵，便于计算  
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  } //???bias_multiplier_和blobs_[1]的shape与实际计算不对应
  
  //乘以scale因子 top blob的shape是M_*N_  
  /*
  for (int _num = 0; _num < top[0]->num(); _num++) { // M_
    for (int _c = 0; _c < top[0]->channels(); _c++){// N_
      int index = top[0]->offset(_num,_c);      
      top_data[index] = top_data[index] * scaleA[_num];
    }
  }
  for (int _c = 0; _c < top[0]->channels(); _c++) { // N_
    for (int _num = 0; _num < top[0]->num(); _num++){// M_
      int index = top[0]->offset(_num,_c);      
      top_data[index] = top_data[index] * scaleA[_c];
    }
  }
  */
}

/*
反向传播
梯度该如何处理？？？
forward过程：sign(I)*sign(W)  系数scaleA*scaleW
backward：scaleA和scaleW的处理：一种对top_diff乘以scale的倒数，但这是一种近似

先暂时不去例会scaleA和scaleW(前向传播也暂时忽略); 增加一个中间的blob用于存储输入二值的更新Blob

*/
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0]) { //权重是否需要更新
    const Dtype* top_diff = top[0]->cpu_diff(); //获取top blob的梯度和bottom数据
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_binary = I_b->cpu_data();    
    // Gradient with respect to weight
    if (transpose_) { //是否转置
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, // K_*M_（M_*K_的转置） M_*N_  K_*N_
          K_, N_, M_,
          (Dtype)1., bottom_binary, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff()); //根据top_diff、bottom_data计算blobs_[0]的梯度
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, //N_*K_
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_binary,
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
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
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
    for (int index = 0; index < bottom[0]->count(); index++) { //判断是否需要回传
      if ( std::abs(bottom_data[index]) > Dtype(1) ) {
        bottom_diff[ index ] = Dtype(0);
      }
    } 
  }
  // 求解器更新参数时，对全精度的权值进行更新
  copyCPUTo(W_buffer, this->blobs_[0]); //恢复全精度的权重用于梯度更新
}
//???在计算梯度的时候更新blob[0]和blob[1]包括原来的更新值 (Dtype)1.
// 注意，此处(Dtype)1., this->blobs_[0]->mutable_gpu_diff()
// 中的(Dtype)1.：使得在一个solver的iteration中的多个iter_size 的梯度没有清零，而得以累加
// 在bottom的反传过程没必要如此，前面的层要依靠得到的梯度更新参数梯度
#ifdef CPU_ONLY
STUB_GPU(   );
#endif

INSTANTIATE_CLASS(BinaryInnerProductLayer);
REGISTER_LAYER_CLASS(BinaryInnerProduct);

}  // namespace caffe
