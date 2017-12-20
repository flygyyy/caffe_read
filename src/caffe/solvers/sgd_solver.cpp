#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
// SGD求解器
namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.  
//    - step: return base_lr * gamma ^ (floor(iter / step))  
//    - exp: return base_lr * gamma ^ iter  
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
//返回当前学习速率，
//根据所选的学习策略和当前迭代进度进行计算；包括固定学习率、步进衰减、指数衰减、倒数衰减、多步衰减、poly衰减、Sigmoid衰减
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy(); //读取学习策略
  if (lr_policy == "fixed") {//根据不同的学习策略，已知迭代次数的基础上计算学习率
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}
//求解预处理
template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  //初始化临时变量，尺寸与训练网络的权值完全一致
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params(); //net_params为整个网络的学习参数
  history_.clear(); //清除历史值、更新、临时值
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) { //net_params.size() 层数
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}
//钳制误差梯度 
//参考链接https://www.zhihu.com/question/29873016
//clip_gradient 的引入是为了处理gradient explosion的问题。当在一次迭代中权重的更新过于迅猛的话，很容易导致loss divergence。
//clip_gradient 的直观作用就是让权重的更新限制在一个合适的范围。
template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients(); //读取设置的钳制梯度值
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff(); //求所有层权重梯度的平方和
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff); //参数矢量的L2范数
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff; //如果sumsq_diff > clip_gradient，则求缩放因子scale_factor 
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}
//应用更新
template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() { //参数更新
  Dtype rate = GetLearningRate();//获取学习率
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {//输出打印信息，设置200iter打印一次
    LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << this->iter_ //迭代次数，学习率
        << ", lr = " << rate;
  }
  ClipGradients();//钳制梯度
  for (int param_id = 0; param_id < this->net_->learnable_params().size();//对训练网络中每个权值blob都做归一化、正则化、计算增量 learnable_params_为可学习的blob数量
       ++param_id) {
    Normalize(param_id);//归一化
    Regularize(param_id);//正则化
    ComputeUpdateValue(param_id, rate);//计算增量
  }
  this->net_->Update();//更新参数
}
//归一化
template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  //以batch_size大小计算梯度和loss，最后再取iter_size次的平均。可以看成iter_size*batch_size次更新一次参数
  //在全连接层和卷积层的反向传播函数，梯度是累积的
  if (this->param_.iter_size() == 1) { return; }//iter_size() = 1的话无需归一化 
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size(); //scale因子
  switch (Caffe::mode()) {
  //void caffe_scal<float>(const int N, const float alpha, float *X) {cblas_sscal(N, alpha, X, 1);}
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,//  功能：X = alpha*X, N： X中element的个数， net_params为要更新的参数
        net_params[param_id]->mutable_cpu_diff()); //计算参数更新的平均梯度
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}
//正则化函数  
//$\nabla w_{ij}=decay*w_{ij}+\nabla w_{ij}$
template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  //正则化项的因子lamda
  Dtype weight_decay = this->param_.weight_decay();//得到solver.prototxt中的weight_decay
  string regularization_type = this->param_.regularization_type();//得到solver.prototxt中的regularization_type
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];//获得每层本地的weight_decay 
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        // L2正则化，引入decay*W项(对w求偏导)
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        // L1正则化，引入decay*sign(W)项
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data()); //temp_[param_id]存储参数的符号
        caffe_axpy(net_params[param_id]->count(),//功能： Y=alpha*X+Y N：为X和Y中element的个数， X为正则化项， Y为求得的梯度
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {  //GPU mode
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}
//更新参数(GPU) 未实现
#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

//计算权值增量
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params(); //可学习参数
  const vector<float>& net_params_lr = this->net_->params_lr(); //学习率
  Dtype momentum = this->param_.momentum(); //动量
  Dtype local_rate = rate * net_params_lr[param_id];//获得当前层的学习速率
  // Compute the update to history, then copy it to the parameter diff.
  // 计算增量存放到history_，之后拷贝到权值的diff域
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    //新的增量=学习速率 x 增量 + 遗忘因子 x 旧的增量
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data()); //上次的权重更新量
    caffe_copy(net_params[param_id]->count(),  //更新本次的权重增量
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY //GPU mode
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

//存储快照
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) { //快照的格式 proto or HDF5
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}
//存储快照为proto格式
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}
//存储快照为HDF5格式
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}
//从proto恢复求解状态
template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}
//从HDF5格式恢复求解状态
template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
