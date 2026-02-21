#pragma once

#include "estimation/AbstractStateEstimator.h"
#include "estimation/MLPNetwork.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <core/misc/Buffer.h>
#include <core/gait/LegLogic.h>
#include <pinocchio/PinocchioInterface.h>
#include <thread>
#include <atomic>

namespace clear {

using namespace rclcpp;

/**
 * @brief 基于机器学习的状态估计器
 *
 * 支持两种模式：
 * 1. 直接学习: x_{k+1} = F_NN(x_k, u_k)
 * 2. 残差学习: x_{k+1} = F_nominal(x_k, u_k) + δ_MLP(x_k, u_k) (推荐)
 *
 * 残差学习的优势:
 * - 保留物理先验知识
 * - MLP只需学习模型误差，数据效率更高
 * - 即使MLP失效，系统退化为名义模型仍可工作
 *
 * 特点：
 * - 支持MLP和RNN架构
 * - 自动归一化输入输出
 * - 与现有MPC框架无缝集成
 * - 实时推理 (< 1ms)
 */
class StateEstimationML : public AbstractStateEstimator {
public:
  /**
   * @brief 构造函数
   * @param nodeHandle ROS节点句柄
   * @param estimator_type 估计器类型 (MLP/RNN)
   */
  explicit StateEstimationML(Node::SharedPtr nodeHandle,
                             EstimatorType estimator_type = EstimatorType::MLP);

  /**
   * @brief 析构函数
   */
  ~StateEstimationML();

  // ========== AbstractStateEstimator接口实现 ==========
  void setup() override;
  std::shared_ptr<vector_t> getQpos() override;
  std::shared_ptr<vector_t> getQvel() override;
  void updateModeSchedule(std::shared_ptr<ModeSchedule> mode_schedule) override;
  void setContactFlag(std::vector<bool> flag) override;
  EstimatorType getType() const override { return estimator_type_; }

private:
  /**
   * @brief 内部循环线程 - 执行状态估计
   */
  void innerLoop();

  /**
   * @brief 使用MLP预测下一步状态
   * @param current_state 当前状态 [position(3), velocity(3), foot_positions(6)]
   * @param control_input 控制输入 [force_x, force_y, force_z] * 2 feet
   * @return 下一步状态
   */
  Eigen::VectorXd predictNextState(const Eigen::VectorXd& current_state,
                                     const Eigen::VectorXd& control_input);

  /**
   * @brief 计算名义动力学模型预测 (仅残差模式使用)
   * @param x 当前状态
   * @param u 控制输入
   * @return 名义模型预测的下一状态
   *
   * 使用简化的线性动力学:
   * x_{k+1} = A * x + B * u + b
   */
  Eigen::VectorXd computeNominalDynamics(const Eigen::VectorXd& x,
                                          const Eigen::VectorXd& u);

  /**
   * @brief IMU回调函数
   */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) const;

  /**
   * @brief 关节状态回调函数
   */
  void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg) const;

  /**
   * @brief 里程计回调函数
   */
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) const;

  /**
   * @brief 从IMU估计角度运动
   */
  void angularMotionEstimate(const sensor_msgs::msg::Imu& imu_data,
                             std::shared_ptr<vector_t> qpos,
                             std::shared_ptr<vector_t> qvel);

  /**
   * @brief 提取当前控制输入 (从机器人模型中获取)
   */
  Eigen::VectorXd extractControlInput();

  // ========== 成员变量 ==========
  Node::SharedPtr nodeHandle_;
  EstimatorType estimator_type_;
  std::string robot_name_;
  scalar_t dt_;
  bool use_odom_;
  bool residual_mode_;  // 残差学习模式标志

  // MLP网络
  std::unique_ptr<MLPNetwork> mlp_network_;

  // 名义动力学模型 (残差模式)
  // x_{k+1} = A * x + B * u + b
  Eigen::MatrixXd A_nominal_;  // 状态转移矩阵 (12x12)
  Eigen::MatrixXd B_nominal_;  // 控制矩阵 (12x6)
  Eigen::VectorXd b_nominal_;  // 偏置向量 (12)
  scalar_t mass_;              // 机器人质量 (kg)
  scalar_t gravity_;           // 重力加速度 (m/s²)

  // Pinocchio机器人模型
  std::unique_ptr<PinocchioInterface> pinocchioInterface_ptr_;
  std::vector<std::string> foot_names_;

  // 状态缓冲区
  Buffer<std::shared_ptr<vector_t>> qpos_ptr_buffer_;
  Buffer<std::shared_ptr<vector_t>> qvel_ptr_buffer_;
  mutable Buffer<sensor_msgs::msg::Imu::SharedPtr> imu_msg_buffer_;
  mutable Buffer<sensor_msgs::msg::JointState::SharedPtr> joint_state_msg_buffer_;
  mutable Buffer<nav_msgs::msg::Odometry::SharedPtr> odom_msg_buffer_;
  Buffer<std::shared_ptr<ModeSchedule>> mode_schedule_buffer_;

  // 状态估计变量
  Eigen::VectorXd x_est_;  // 当前状态估计 (12维)

  // ROS订阅和发布
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joints_state_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_est_publisher_;

  // 线程控制
  std::thread inner_loop_thread_;
  Buffer<bool> run_;
  std::atomic<bool> initialized_{false};

  // 接触标志
  std::vector<bool> cflag_;
};

} // namespace clear
