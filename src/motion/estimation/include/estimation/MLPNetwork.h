#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace clear {

/**
 * @brief 激活函数类型
 */
enum class ActivationType {
  ReLU,
  Tanh,
  Sigmoid,
  Linear
};

/**
 * @brief 简单的MLP网络实现 (基于Eigen)
 *
 * 轻量级MLP实现，支持：
 * - 任意层数和神经元数量
 * - ReLU/Tanh/Sigmoid激活函数
 * - 输入输出归一化
 * - 模型参数加载/保存
 */
class MLPNetwork {
public:
  /**
   * @brief 构造函数 - 从配置文件加载网络结构
   * @param config_file YAML配置文件路径
   */
  explicit MLPNetwork(const std::string& config_file);

  /**
   * @brief 构造函数 - 手动指定网络结构
   * @param input_dim 输入维度
   * @param hidden_dims 隐藏层维度列表
   * @param output_dim 输出维度
   * @param activation 激活函数类型
   */
  MLPNetwork(int input_dim,
             const std::vector<int>& hidden_dims,
             int output_dim,
             ActivationType activation = ActivationType::ReLU);

  /**
   * @brief 前向传播
   * @param input 输入向量
   * @return 输出向量
   */
  Eigen::VectorXd forward(const Eigen::VectorXd& input);

  /**
   * @brief 加载模型参数
   * @param model_file 模型文件路径 (.txt格式)
   * @return 是否加载成功
   */
  bool loadModel(const std::string& model_file);

  /**
   * @brief 保存模型参数
   * @param model_file 模型文件路径
   * @return 是否保存成功
   */
  bool saveModel(const std::string& model_file) const;

  /**
   * @brief 设置输入归一化参数
   * @param mean 均值向量
   * @param std 标准差向量
   */
  void setInputNormalization(const Eigen::VectorXd& mean, const Eigen::VectorXd& std);

  /**
   * @brief 设置输出归一化参数
   * @param mean 均值向量
   * @param std 标准差向量
   */
  void setOutputNormalization(const Eigen::VectorXd& mean, const Eigen::VectorXd& std);

  /**
   * @brief 获取输入维度
   */
  int getInputDim() const { return input_dim_; }

  /**
   * @brief 获取输出维度
   */
  int getOutputDim() const { return output_dim_; }

private:
  /**
   * @brief 应用激活函数
   * @param x 输入向量
   * @return 激活后的向量
   */
  Eigen::VectorXd activate(const Eigen::VectorXd& x) const;

  /**
   * @brief 标准化输入
   */
  Eigen::VectorXd normalizeInput(const Eigen::VectorXd& input) const;

  /**
   * @brief 反标准化输出
   */
  Eigen::VectorXd denormalizeOutput(const Eigen::VectorXd& output) const;

  int input_dim_;
  int output_dim_;
  std::vector<int> hidden_dims_;
  ActivationType activation_;

  // 网络权重和偏置
  std::vector<Eigen::MatrixXd> weights_;  // W_i: hidden_dims_[i] × prev_dim
  std::vector<Eigen::VectorXd> biases_;   // b_i: hidden_dims_[i]

  // 归一化参数
  bool use_input_norm_ = false;
  bool use_output_norm_ = false;
  Eigen::VectorXd input_mean_, input_std_;
  Eigen::VectorXd output_mean_, output_std_;
};

} // namespace clear
