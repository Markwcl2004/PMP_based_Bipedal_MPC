#include "estimation/MLPNetwork.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace clear {

// ========== 构造函数 ==========

MLPNetwork::MLPNetwork(const std::string& config_file) {
  YAML::Node config = YAML::LoadFile(config_file);

  input_dim_ = config["input_dim"].as<int>();
  output_dim_ = config["output_dim"].as<int>();
  hidden_dims_ = config["hidden_dims"].as<std::vector<int>>();

  std::string act_str = config["activation"].as<std::string>();
  if (act_str == "relu") activation_ = ActivationType::ReLU;
  else if (act_str == "tanh") activation_ = ActivationType::Tanh;
  else if (act_str == "sigmoid") activation_ = ActivationType::Sigmoid;
  else activation_ = ActivationType::Linear;

  // 初始化权重和偏置 (随机初始化，之后通过loadModel加载)
  int prev_dim = input_dim_;
  for (int hidden_dim : hidden_dims_) {
    weights_.emplace_back(Eigen::MatrixXd::Random(hidden_dim, prev_dim) * 0.1);
    biases_.emplace_back(Eigen::VectorXd::Zero(hidden_dim));
    prev_dim = hidden_dim;
  }
  // 输出层
  weights_.emplace_back(Eigen::MatrixXd::Random(output_dim_, prev_dim) * 0.1);
  biases_.emplace_back(Eigen::VectorXd::Zero(output_dim_));

  // 加载归一化参数 (如果存在)
  if (config["normalization"]) {
    use_input_norm_ = config["normalization"]["use_input_norm"].as<bool>(false);
    use_output_norm_ = config["normalization"]["use_output_norm"].as<bool>(false);

    if (use_input_norm_) {
      auto mean_vec = config["normalization"]["input_mean"].as<std::vector<double>>();
      auto std_vec = config["normalization"]["input_std"].as<std::vector<double>>();
      input_mean_ = Eigen::Map<Eigen::VectorXd>(mean_vec.data(), mean_vec.size());
      input_std_ = Eigen::Map<Eigen::VectorXd>(std_vec.data(), std_vec.size());
    }

    if (use_output_norm_) {
      auto mean_vec = config["normalization"]["output_mean"].as<std::vector<double>>();
      auto std_vec = config["normalization"]["output_std"].as<std::vector<double>>();
      output_mean_ = Eigen::Map<Eigen::VectorXd>(mean_vec.data(), mean_vec.size());
      output_std_ = Eigen::Map<Eigen::VectorXd>(std_vec.data(), std_vec.size());
    }
  }

  // 加载模型文件 (如果指定)
  if (config["model_path"]) {
    std::string model_path = config["model_path"].as<std::string>();
    if (!loadModel(model_path)) {
      std::cerr << "[MLPNetwork] Warning: Failed to load model from " << model_path << std::endl;
    }
  }
}

MLPNetwork::MLPNetwork(int input_dim,
                       const std::vector<int>& hidden_dims,
                       int output_dim,
                       ActivationType activation)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      hidden_dims_(hidden_dims),
      activation_(activation) {

  // 初始化权重和偏置
  int prev_dim = input_dim_;
  for (int hidden_dim : hidden_dims_) {
    weights_.emplace_back(Eigen::MatrixXd::Random(hidden_dim, prev_dim) * 0.1);
    biases_.emplace_back(Eigen::VectorXd::Zero(hidden_dim));
    prev_dim = hidden_dim;
  }
  // 输出层
  weights_.emplace_back(Eigen::MatrixXd::Random(output_dim_, prev_dim) * 0.1);
  biases_.emplace_back(Eigen::VectorXd::Zero(output_dim_));
}

// ========== 前向传播 ==========

Eigen::VectorXd MLPNetwork::forward(const Eigen::VectorXd& input) {
  if (input.size() != input_dim_) {
    throw std::runtime_error("[MLPNetwork] Input dimension mismatch: expected " +
                             std::to_string(input_dim_) + ", got " +
                             std::to_string(input.size()));
  }

  // 标准化输入
  Eigen::VectorXd x = use_input_norm_ ? normalizeInput(input) : input;

  // 前向传播 - 隐藏层
  for (size_t i = 0; i < hidden_dims_.size(); ++i) {
    x = weights_[i] * x + biases_[i];
    x = activate(x);
  }

  // 输出层 (线性激活)
  x = weights_.back() * x + biases_.back();

  // 反标准化输出
  return use_output_norm_ ? denormalizeOutput(x) : x;
}

// ========== 激活函数 ==========

Eigen::VectorXd MLPNetwork::activate(const Eigen::VectorXd& x) const {
  switch (activation_) {
    case ActivationType::ReLU:
      return x.cwiseMax(0.0);
    case ActivationType::Tanh:
      return x.array().tanh().matrix();
    case ActivationType::Sigmoid:
      return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    case ActivationType::Linear:
      return x;
    default:
      return x;
  }
}

// ========== 归一化 ==========

Eigen::VectorXd MLPNetwork::normalizeInput(const Eigen::VectorXd& input) const {
  return (input - input_mean_).cwiseQuotient(input_std_);
}

Eigen::VectorXd MLPNetwork::denormalizeOutput(const Eigen::VectorXd& output) const {
  return output.cwiseProduct(output_std_) + output_mean_;
}

void MLPNetwork::setInputNormalization(const Eigen::VectorXd& mean, const Eigen::VectorXd& std) {
  input_mean_ = mean;
  input_std_ = std;
  use_input_norm_ = true;
}

void MLPNetwork::setOutputNormalization(const Eigen::VectorXd& mean, const Eigen::VectorXd& std) {
  output_mean_ = mean;
  output_std_ = std;
  use_output_norm_ = true;
}

// ========== 模型加载/保存 ==========

bool MLPNetwork::loadModel(const std::string& model_file) {
  std::ifstream file(model_file);
  if (!file.is_open()) {
    std::cerr << "[MLPNetwork] Failed to open model file: " << model_file << std::endl;
    return false;
  }

  try {
    // 读取网络结构
    int input_dim, output_dim, num_hidden;
    file >> input_dim >> output_dim >> num_hidden;

    if (input_dim != input_dim_ || output_dim != output_dim_) {
      std::cerr << "[MLPNetwork] Model dimension mismatch!" << std::endl;
      return false;
    }

    // 读取隐藏层维度
    std::vector<int> hidden_dims(num_hidden);
    for (int& dim : hidden_dims) file >> dim;

    // 读取权重和偏置
    for (size_t i = 0; i < weights_.size(); ++i) {
      int rows = weights_[i].rows();
      int cols = weights_[i].cols();

      // 读取权重矩阵
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          file >> weights_[i](r, c);
        }
      }

      // 读取偏置向量
      for (int r = 0; r < rows; ++r) {
        file >> biases_[i](r);
      }
    }

    std::cout << "[MLPNetwork] Model loaded successfully from " << model_file << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MLPNetwork] Error loading model: " << e.what() << std::endl;
    return false;
  }
}

bool MLPNetwork::saveModel(const std::string& model_file) const {
  std::ofstream file(model_file);
  if (!file.is_open()) {
    std::cerr << "[MLPNetwork] Failed to create model file: " << model_file << std::endl;
    return false;
  }

  try {
    // 写入网络结构
    file << input_dim_ << " " << output_dim_ << " " << hidden_dims_.size() << "\n";
    for (int dim : hidden_dims_) file << dim << " ";
    file << "\n";

    // 写入权重和偏置
    for (size_t i = 0; i < weights_.size(); ++i) {
      const auto& W = weights_[i];
      const auto& b = biases_[i];

      // 写入权重矩阵
      for (int r = 0; r < W.rows(); ++r) {
        for (int c = 0; c < W.cols(); ++c) {
          file << W(r, c) << " ";
        }
        file << "\n";
      }

      // 写入偏置向量
      for (int r = 0; r < b.size(); ++r) {
        file << b(r) << " ";
      }
      file << "\n";
    }

    std::cout << "[MLPNetwork] Model saved successfully to " << model_file << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MLPNetwork] Error saving model: " << e.what() << std::endl;
    return false;
  }
}

} // namespace clear
