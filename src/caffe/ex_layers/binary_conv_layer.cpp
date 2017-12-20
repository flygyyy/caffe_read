#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/ex_layers/binary_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
// 二值卷积层 实现sign(I)和sign(W)的卷积
namespace caffe {

#ifdef USE_CUDNN
#define CUDNN_STREAMS_PER_GROUP 3
#endif
/*
层初始化函数
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  // 配置核尺寸、padding、步长、输入
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();//是否使用n维im2col方法计算
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1; //1+1为2
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;// 图像的维度2或者3
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1); //bottom维度向量
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));//spatial维度向量

  // 设置卷积核尺寸
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }

  // 设置步长
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // 设置pad
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // 特殊情况， im2col是步长为1的1X1的卷积
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) { //判断是否为1X1的卷积
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // 配置输出channel和group
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_); // channel_axis_输入图像的第几个轴是通道  获取bottom的channel
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) { //维度是否反转
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // 处理参数：权重和偏置 
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_; //输出channel的数目
  weight_shape[1] = conv_in_channels_ / group_; //分组
  
  // weight_shape = [conv_out_channels_, conv_in_channels_/group, weight_h, weight_w]
  // conv_input_shape_data = [inchannel, in_height, in width]
  // kernel_shape_=[kernel_height, kernel_width]
  
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) { //是否存在偏置项
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // 初始化权重和偏置
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

//CUDNN  cudnn的卷积层实现
#ifdef USE_CUDNN
// Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* cudnn_kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = cudnn_kernel_shape_data[0];
  const int kernel_w = cudnn_kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
#endif
//end of CUDA

  // For Binary
  this->W_b = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->W_buffer = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  this->W_b->ReshapeLike(*(this->blobs_[0]));
  this->W_buffer->ReshapeLike(*(this->blobs_[0]));
}

/*
变形
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

//CUDNN
#ifdef USE_CUDNN
CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
#endif
//cudnn实现
  this->W_b->ReshapeLike(*(this->blobs_[0]));
  this->W_buffer->ReshapeLike(*(this->blobs_[0]));
}

/*
计算输出size
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data(); //读取相关数据
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear(); //清除output_shape_
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}
/*
前向推理
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  binarizeCPUTo(this->blobs_[0], W_b); //得到二值化权重W_b，sign(W)乘以scale之后的权重
  copyCPUTo(this->blobs_[0], W_buffer);//blobs_[0]暂存到W_buffer
  copyCPUTo(W_b, this->blobs_[0]);//复制二值权重
 
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) { //输入bottom blob的数目
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {//num_ = batchsize 根据每个样本的输入、权重得到卷积输出
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,  //  bottom_dim_:conv_in_channel X in_height X in_width
          top_data + n * this->top_dim_); //top_dim_:conv_out_channel X out_height X out_width
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}
/*
反向计算
//top_dim_:conv_out_channel X out_height X out_width
//bottom_dim_:conv_in_channel X in_height X in_width
以下三个函数用于计算梯度：
backward_cpu_bias
weight_cpu_gemm
backward_cpu_gemm
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = this->blobs_[0]->cpu_data(); //二值权重
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {//size是否可理解为num
    const Dtype* top_diff = top[i]->cpu_diff(); //读取top_diff和bottom_data用于计算梯度
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) { //计算bias_diff
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) { //在一个batch上进行计算，求均值
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) { //在一个batch上进行计算，求均值
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) { //计算weight_diff
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) { //计算bottom_diff
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  copyCPUTo(W_buffer, this->blobs_[0]); //恢复全精度的权重用于梯度更新
}

/*
对权重进行二值化处理
*/ 
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::binarizeCPUTo(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb) {
  //rand_vec_.ReshapeLike(*weights);
  //caffe_rng_uniform(rand_vec_.count(), 0, 1, rand_vec_.mutable_cpu_data());
  //const Dtype *prob = rand_vec_.cpu_data();
  #define sign(x) ((x)>=0?1:-1)
  const int div = weights->count() / weights->num(); //总数除以输出个数
  std::vector<Dtype> A(weights->num(), 0); //输出通道的个数num()，存储权重scale
  for (int num = 0; num < weights->num(); num++) {
    //Dtype det = hard_sigmoid(weights->cpu_data()[index]);
    for (int _c = 0; _c < weights->channels(); _c++)
    for (int _h = 0; _h < weights->height(); _h++)
    for (int _w = 0; _w < weights->width(); _w++)
      A[num] += std::abs(weights->data_at(num, _c, _h, _w)) / Dtype(div); //计算L1范数并除以元素个数，得到weight的scale因子， 一共有num_个scale
  }
  for (int index = 0; index < weights->count(); index++) {
    const int num = index / div;
    wb->mutable_cpu_data()[index] = A[num] * sign(weights->cpu_data()[index]); //二值化权重sign(W)乘以scale  AAAI2017的论文中将scale提取到PReLU层进行处理，wb得到的是float类型，无法用于加速
  }
}

/*
前向计算(权重)
输入，权重，输出，是否跳过转换
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) { 
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) { 
      // 如果没有1x1卷积，也没有skip_im2col    
      // 则使用conv_im2col_cpu对使用卷积核滑动过程中的每一个kernel大小的图像块    
      // 变成一个列向量，形成一个height=kernel_dim_    
      // width = 卷积后图像heght*卷积后图像width  
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    // 分组分别进行计算    
    // conv_out_channels_ / group_是每个卷积组的输出的channel    
    // kernel_dim_ = input channels per-group x kernel height x kernel width    
    // 计算的是output[output_offset_ * g] =    
    // weights[weight_offset_ * g] X col_buff[col_offset_ * g]    
    // weights的形状是 [conv_out_channel x kernel_dim_]    
    // col_buff的形状是[kernel_dim_ x (卷积后图像高度乘以卷积后图像宽度)]    
    // 所以output的形状自然就是conv_out_channel X (卷积后图像高度乘以卷积后图像宽度)  
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
/*
input*weight后累加bias
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  // output = bias * bias_multiplier_    
  // num_output 与 conv_out_channel是一样的    
  // num_output_ X out_spatial_dim_ = num_output_ X 1   1 X out_spatial_dim_    
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}
/*
反向传播计算关于bottom data的导数
backward_cpu_gemm(top_diff + n * this->top_dim_, weight, bottom_diff + n * this->bottom_dim_);
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}
/*
weight反向传播
weight_cpu_gemm(bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff);
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}
/*
bias反向传播
backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_)
*/
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}


#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);
REGISTER_LAYER_CLASS(BinaryConvolution);

}  // namespace caffe
