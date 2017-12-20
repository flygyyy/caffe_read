#include <vector>

#include "caffe/filler.hpp"
#include "caffe/ex_layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"


//二值化全连接层
namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();

  //对输入进行二值化
  std::vector<Dtype> scaleA(bottom[0]->num(), 0); //输出通道的个数num()，存储权重scale 
  int index = 0; 
  //对输入和权重进行二值化处理，得到sign(I)和sign(W)，并得到scale因子
  //判断输入数据的格式
  if(bottom[0]->channels() == K_){//shape为2
    for (int _num = 0; index < bottom[0]->num(); _num++ ) {
      for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
        index = bottom[0]->offset(_num,_c);        
        scaleA[ _num ] += std::abs(bottom[0]->gpu_data()[index]) / Dtype(K_);
        I_b->mutable_gpu_data()[index] = sign(bottom[0]->gpu_data()[index]);              
      }
    }
  }
  else { //shape为4
    for (int _num = 0; index < bottom[0]->num(); _num++ ) {
      for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
        for (int _h = 0; _h < bottom[0]->height(); _h++ ) {
          for (int _w = 0; _w < bottom[0]->width(); _w++ ) {
            index = bottom[0]->offset(index,_c,_h,_w);            
            scaleA[ _num ] += std::abs(bottom[0]->gpu_data()[index]) / K_;
            I_b->mutable_gpu_data()[index] = sign(bottom[0]->gpu_data()[index]);              
          }
        }
      }
    }
  }
  //计算得到二值blob I_b
  //copyGPUTo(bottom[0], I_buffer);//全精度的权重blobs_[0]暂存到W_buffer
  //copyGPUTo(I_b, bottom[0]);//复制二值权重到权重参数

  // 对权重进行二值化 权重的shape就是2 N_*K_
  std::vector<Dtype> scaleW(this->blobs_[0]->num(), 0); //输出通道的个数num()，存储权重scale    
  for (int _num = 0; _num < this->blobs_[0]->num(); _num++) {
    for (int _c = 0; _c < this->blobs_[0]->channels(); _c++){
      index = this->blobs_[0]->offset(_num,_c);
      scaleW[_num] += std::abs(this->blobs_[0]->gpu_data()[index]) / Dtype(K_); //计算L1范数并除以元素个数，得到weight的scale因子， 一共有num_个scale
      W_b->mutable_gpu_data()[index] = sign(this->blobs_[0]->gpu_data()[index]);
    }
  }
  copyGPUTo(this->blobs_[0], W_buffer);//全精度的权重blobs_[0]暂存到W_buffer
  copyGPUTo(W_b, this->blobs_[0]);//复制二值权重到权重参数

  //const Dtype* bottom_data = bottom[0]->gpu_data(); //输入数据， 常量  
  const Dtype* bottom_binary = I_b->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();//读取权重数据

  caffe_gpu_gemm<Dtype>(CblasNoTrans,
                        transpose_ ? CblasNoTrans : CblasTrans,
                        M_, N_, K_, (Dtype)1.,
                        bottom_binary, weight, (Dtype)0., top_data);
  if (bias_term_)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.gpu_data(),
                          this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    //const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_binary = I_b->gpu_data();    

    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_binary, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_binary,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
    for (int index = 0; index < bottom[0]->count(); index++) { //判断是否需要回传
      if ( std::abs(bottom_data[index]) > Dtype(1) ) {
        bottom_diff[ index ] = Dtype(0);
      }
    } 

  }
  copyGPUTo(W_buffer, this->blobs_[0]); //恢复全精度的权重用于梯度更新, 输入bottom blob保持二值化

}
INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);

}  // namespace caffe
