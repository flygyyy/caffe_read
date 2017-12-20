#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"
//计算elementwise操作，例如多个输入的点的和或者乘积
namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0   // coeff blob-wise coefficient for SUM operation
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) << //每个bottom blob对应一个coeff
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) << // sum操作
      "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i); // 读取coeff因子
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}
/*
reshpe
*/
template <typename Dtype>
void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) { //检查各个bottom blob的形状是否一致
    CHECK(bottom[0]->shape() == bottom[i]->shape())
        << "bottom[0]: " << bottom[0]->shape_string()
        << ", bottom[" << i << "]: " << bottom[i]->shape_string();
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.  如果是Max操作初始化 向量索引部分max_idx_
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}
/*
前向推理
*/
template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD: //乘法操作
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM: //加法操作
    caffe_set(count, Dtype(0), top_data); //初始化top_data为0
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data); //Y=alpha*X+Y  coeffs_为输入blob的系数
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX: // 最大值操作
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask); //mask数据初始化为-1
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid 记录max值属于哪一个bottom blob
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}
/*
反向传播
*/
template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) { //数据反向传播
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff); //bottom_diff先赋初值，以防以后相乘为0
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff, // 所有bottom数据相乘
                        bottom_diff);  
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff); //直接用top data除以 bottom data就是剩余bottom的乘积
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM: //SUM
        if (coeffs_[i] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff); //有coeff_因子
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];  //如果bottom 的数值为最大值，diff为top diff；否则为0
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);
REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
