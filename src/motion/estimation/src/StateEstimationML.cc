#include "estimation/StateEstimationML.h"
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>

namespace clear {

StateEstimationML::StateEstimationML(Node::SharedPtr nodeHandle,
                                      EstimatorType estimator_type)
    : nodeHandle_(nodeHandle), estimator_type_(estimator_type) {

  const std::string config_file = nodeHandle_->get_parameter("/config_file")
                                      .get_parameter_value()
                                      .get<std::string>();

  auto config = YAML::LoadFile(config_file);

  robot_name_ = config["model"]["name"].as<std::string>();
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
              "Initializing ML State Estimator for robot: %s", robot_name_.c_str());
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
              "Estimator type: %s", getTypeName().c_str());

  std::string model_package = config["model"]["package"].as<std::string>();
  std::string urdf =
      ament_index_cpp::get_package_share_directory(model_package) +
      config["model"]["urdf"].as<std::string>();

  pinocchioInterface_ptr_ = std::make_unique<PinocchioInterface>(urdf.c_str());
  foot_names_ = config["model"]["foot_names"].as<std::vector<std::string>>();
  dt_ = config["estimation"]["dt"].as<scalar_t>();
  use_odom_ = config["estimation"]["use_odom"].as<bool>(false);

  // 机器人物理参数
  mass_ = config["model"]["mass"].as<scalar_t>(12.0);  // 默认12kg
  gravity_ = 9.81;

  for (const auto& name : foot_names_) {
    RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"), "Foot name: %s", name.c_str());
    cflag_.emplace_back(true);
  }

  // 加载MLP网络配置
  std::string ml_config_file = config["estimation"]["ml_config"].as<std::string>("");
  if (ml_config_file.empty()) {
    // 默认配置
    ml_config_file = ament_index_cpp::get_package_share_directory(model_package) +
                     "/config/mlp_estimator.yaml";
  }

  if (estimator_type_ == EstimatorType::MLP) {
    try {
      mlp_network_ = std::make_unique<MLPNetwork>(ml_config_file);
      RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
                  "MLP network loaded successfully");
      RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
                  "Input dim: %d, Output dim: %d",
                  mlp_network_->getInputDim(), mlp_network_->getOutputDim());

      // 从MLP配置中读取残差模式标志
      auto ml_config = YAML::LoadFile(ml_config_file);
      residual_mode_ = ml_config["residual_mode"].as<bool>(false);
      RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
                  "Residual mode: %s", residual_mode_ ? "ENABLED" : "DISABLED");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("StateEstimationML"),
                   "Failed to load MLP network: %s", e.what());
      RCLCPP_WARN(rclcpp::get_logger("StateEstimationML"),
                  "Using random initialization");
      // 使用默认结构
      mlp_network_ = std::make_unique<MLPNetwork>(
          18,                      // input: state(12) + control(6)
          std::vector<int>{128, 64, 32},  // hidden layers
          12,                      // output: next_state(12)
          ActivationType::ReLU);
      residual_mode_ = false;
    }
  }

  // ROS订阅和发布
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);
  std::string topic_prefix = config["global"]["topic_prefix"].as<std::string>();

  std::string imu_topic = config["global"]["topic_names"]["imu"].as<std::string>();
  imu_subscription_ = nodeHandle_->create_subscription<sensor_msgs::msg::Imu>(
      topic_prefix + imu_topic, qos,
      std::bind(&StateEstimationML::imuCallback, this, std::placeholders::_1));

  std::string joints_topic =
      config["global"]["topic_names"]["joints_state"].as<std::string>();
  joints_state_subscription_ =
      nodeHandle_->create_subscription<sensor_msgs::msg::JointState>(
          topic_prefix + joints_topic, qos,
          std::bind(&StateEstimationML::jointCallback, this,
                    std::placeholders::_1));

  std::string odom_topic = config["global"]["topic_names"]["odom"].as<std::string>();
  odom_subscription_ =
      nodeHandle_->create_subscription<nav_msgs::msg::Odometry>(
          topic_prefix + odom_topic, qos,
          std::bind(&StateEstimationML::odomCallback, this,
                    std::placeholders::_1));

  odom_est_publisher_ = nodeHandle_->create_publisher<nav_msgs::msg::Odometry>(
      topic_prefix + odom_topic + "_est", qos);

  inner_loop_thread_ = std::thread(&StateEstimationML::innerLoop, this);
  run_.push(true);
  setup();
}

StateEstimationML::~StateEstimationML() {
  run_.push(false);
  if (inner_loop_thread_.joinable()) {
    inner_loop_thread_.join();
  }
}

void StateEstimationML::setup() {
  x_est_ = Eigen::VectorXd::Zero(12);
  x_est_(2) = 0.5;  // 初始z位置 (高度)

  // 初始化名义动力学模型矩阵 (残差模式)
  // 状态定义: [pos(3), vel(3), rpy(3), omega(3)]
  // 动力学: x_{k+1} = A * x + B * u + b

  A_nominal_ = Eigen::MatrixXd::Identity(12, 12);
  // 位置更新: p_{k+1} = p_k + dt * v_k
  A_nominal_.block<3, 3>(0, 3) = dt_ * Eigen::Matrix3d::Identity();

  B_nominal_ = Eigen::MatrixXd::Zero(12, 6);
  // 速度更新: v_{k+1} = v_k + dt/m * f_total
  // 简化: 只考虑z方向的力对速度的影响
  B_nominal_.block<3, 3>(3, 0) = (dt_ / mass_) * Eigen::Matrix3d::Identity();
  B_nominal_.block<3, 3>(3, 3) = (dt_ / mass_) * Eigen::Matrix3d::Identity();

  b_nominal_ = Eigen::VectorXd::Zero(12);
  // 重力影响: vz_{k+1} = vz_k - dt * g
  b_nominal_(5) = -dt_ * gravity_;

  RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"),
              "Nominal dynamics initialized: dt=%.4f, mass=%.2f, gravity=%.2f",
              dt_, mass_, gravity_);

  initialized_ = true;
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationML"), "Setup completed");
}

std::shared_ptr<vector_t> StateEstimationML::getQpos() {
  return qpos_ptr_buffer_.get();
}

std::shared_ptr<vector_t> StateEstimationML::getQvel() {
  return qvel_ptr_buffer_.get();
}

void StateEstimationML::updateModeSchedule(std::shared_ptr<ModeSchedule> mode_schedule) {
  mode_schedule_buffer_.push(mode_schedule);
}

void StateEstimationML::setContactFlag(std::vector<bool> flag) {
  cflag_ = flag;
}

Eigen::VectorXd StateEstimationML::predictNextState(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& control_input) {

  if (!mlp_network_) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("StateEstimationML"),
                         *nodeHandle_->get_clock(), 1000,
                         "MLP network not initialized, returning current state");
    return current_state;
  }

  // 构建输入: [state(12), control(6)]
  Eigen::VectorXd input(18);
  input << current_state, control_input;

  if (residual_mode_) {
    // 残差模式: x_{k+1} = F_nominal(x, u) + δ_MLP(x, u)
    Eigen::VectorXd x_nominal = computeNominalDynamics(current_state, control_input);
    Eigen::VectorXd delta = mlp_network_->forward(input);
    return x_nominal + delta;
  } else {
    // 直接学习模式: x_{k+1} = F_MLP(x, u)
    return mlp_network_->forward(input);
  }
}

Eigen::VectorXd StateEstimationML::computeNominalDynamics(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  // 简化的线性动力学模型:
  // x_{k+1} = A * x + B * u + b
  return A_nominal_ * x + B_nominal_ * u + b_nominal_;
}

Eigen::VectorXd StateEstimationML::extractControlInput() {
  // TODO: 从MPC或机器人状态中提取控制输入
  // 目前返回零输入
  return Eigen::VectorXd::Zero(6);
}

void StateEstimationML::angularMotionEstimate(
    const sensor_msgs::msg::Imu& imu_data,
    std::shared_ptr<vector_t> qpos,
    std::shared_ptr<vector_t> qvel) {

  const auto& quat_imu = imu_data.orientation;
  (*qpos).segment(3, 4) << quat_imu.x, quat_imu.y, quat_imu.z, quat_imu.w;
  (*qvel).segment(3, 3) << imu_data.angular_velocity.x,
                           imu_data.angular_velocity.y,
                           imu_data.angular_velocity.z;
}

void StateEstimationML::innerLoop() {
  rclcpp::Rate loop_rate(1.0 / dt_);

  const scalar_t t0 = nodeHandle_->now().seconds();
  while (rclcpp::ok() && run_.get()) {
    // 确保所有传感器数据已到达
    if (imu_msg_buffer_.get().get() == nullptr ||
        joint_state_msg_buffer_.get().get() == nullptr ||
        (use_odom_ && odom_msg_buffer_.get().get() == nullptr)) {
      loop_rate.sleep();
      continue;
    }

    sensor_msgs::msg::Imu::SharedPtr imu_msg = imu_msg_buffer_.get();
    sensor_msgs::msg::JointState::SharedPtr joint_state_msg =
        joint_state_msg_buffer_.get();
    nav_msgs::msg::Odometry::SharedPtr odom_msg = odom_msg_buffer_.get();

    const pin::Model& model = pinocchioInterface_ptr_->getModel();
    std::shared_ptr<vector_t> qpos_ptr = std::make_shared<vector_t>(model.nq);
    qpos_ptr->setZero();
    std::shared_ptr<vector_t> qvel_ptr = std::make_shared<vector_t>(model.nv);
    qvel_ptr->setZero();

    // 更新关节状态
    for (size_t k = 0; k < joint_state_msg->name.size(); k++) {
      if (static_cast<int>(k) < model.njoints &&
          model.existJointName(joint_state_msg->name[k])) {
        pin::Index id = model.getJointId(joint_state_msg->name[k]) - 2;
        (*qpos_ptr)[id + 7] = joint_state_msg->position[k];
        (*qvel_ptr)[id + 6] = joint_state_msg->velocity[k];
      }
    }

    if (use_odom_) {
      // 使用里程计数据
      const auto& orientation = odom_msg->pose.pose.orientation;
      const auto& position = odom_msg->pose.pose.position;
      const auto& vel = odom_msg->twist.twist.linear;
      const auto& ang_vel = odom_msg->twist.twist.angular;

      Eigen::Quaternion<scalar_t> quat(orientation.w, orientation.x,
                                       orientation.y, orientation.z);
      matrix3_t rot = quat.toRotationMatrix();
      (*qpos_ptr).head(3) << position.x, position.y, position.z;
      (*qvel_ptr).head(3) << vel.x, vel.y, vel.z;
      (*qvel_ptr).head(3) = rot.transpose() * (*qvel_ptr).head(3);

      (*qpos_ptr).segment(3, 4) << orientation.x, orientation.y, orientation.z,
          orientation.w;
      (*qvel_ptr).segment(3, 3) << ang_vel.x, ang_vel.y, ang_vel.z;
    } else {
      // 使用ML估计器
      angularMotionEstimate(*imu_msg, qpos_ptr, qvel_ptr);

      // 提取控制输入
      Eigen::VectorXd u = extractControlInput();

      // 使用MLP预测下一步状态
      Eigen::VectorXd x_current = x_est_;
      Eigen::VectorXd x_next = predictNextState(x_current, u);

      // 更新状态估计
      x_est_ = x_next;

      // 更新位姿和速度
      (*qpos_ptr).head(3) = x_est_.head(3);  // 位置
      (*qvel_ptr).head(3) = x_est_.segment(3, 3);  // 速度
    }

    if (nodeHandle_->now().seconds() - t0 > 0.3) {
      qpos_ptr_buffer_.push(qpos_ptr);
      qvel_ptr_buffer_.push(qvel_ptr);
    }

    // 发布里程计消息
    auto message = nav_msgs::msg::Odometry();
    {
      message.header.frame_id = robot_name_;
      message.header.stamp = nodeHandle_->now();
      message.pose.pose.position.x = qpos_ptr->x();
      message.pose.pose.position.y = qpos_ptr->y();
      message.pose.pose.position.z = qpos_ptr->z();
      message.pose.pose.orientation.w = (*qpos_ptr)[6];
      message.pose.pose.orientation.x = (*qpos_ptr)[3];
      message.pose.pose.orientation.y = (*qpos_ptr)[4];
      message.pose.pose.orientation.z = (*qpos_ptr)[5];

      Eigen::Quaternion<scalar_t> quat((*qpos_ptr)[6], (*qpos_ptr)[3],
                                       (*qpos_ptr)[4], (*qpos_ptr)[5]);
      vector3_t vel_world = quat.toRotationMatrix() * qvel_ptr->head(3);
      message.twist.twist.linear.x = vel_world.x();
      message.twist.twist.linear.y = vel_world.y();
      message.twist.twist.linear.z = vel_world.z();
      message.twist.twist.angular.x = (*qvel_ptr)[3];
      message.twist.twist.angular.y = (*qvel_ptr)[4];
      message.twist.twist.angular.z = (*qvel_ptr)[5];
    }
    odom_est_publisher_->publish(message);

    loop_rate.sleep();
  }
}

void StateEstimationML::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name_) {
    imu_msg_buffer_.push(msg);
  }
}

void StateEstimationML::jointCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name_) {
    joint_state_msg_buffer_.push(msg);
  }
}

void StateEstimationML::odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name_) {
    odom_msg_buffer_.push(msg);
  }
}

} // namespace clear
