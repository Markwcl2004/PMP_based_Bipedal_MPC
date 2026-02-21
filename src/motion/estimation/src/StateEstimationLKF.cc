#include "estimation/StateEstimationLKF.h"

#include <core/gait/LegLogic.h>
#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

/*
基于线性卡尔曼滤波（LKF）的状态估计，估计机器人基座位置、速度和关节状态。

innerLoop: 内部循环，定期更新状态估计。
linearMotionEstimate: 线性运动估计。
angularMotionEstimate: 角运动估计。
imuCallback: 处理 IMU 数据回调。
*/

namespace clear {

StateEstimationLKF::StateEstimationLKF(Node::SharedPtr nodeHandle)
    : nodeHandle_(nodeHandle) {
  const std::string config_file_ = nodeHandle_->get_parameter("/config_file")
                                       .get_parameter_value()
                                       .get<std::string>();

  auto config_ = YAML::LoadFile(config_file_);

  robot_name = config_["model"]["name"].as<std::string>();
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"), "robot_name: %s",
              robot_name.c_str());

  std::string model_package = config_["model"]["package"].as<std::string>();
  std::string urdf =
      ament_index_cpp::get_package_share_directory(model_package) +
      config_["model"]["urdf"].as<std::string>();
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"), "model file: %s",
              urdf.c_str());
  //从配置文件中读取脚部名称、时间步长 dt_ 和是否使用里程计 use_odom_，并打印日志信息
  pinocchioInterface_ptr = std::make_unique<PinocchioInterface>(urdf.c_str());
  foot_names = config_["model"]["foot_names"].as<std::vector<std::string>>();
  dt_ = config_["estimation"]["dt"].as<scalar_t>();
  use_odom_ = config_["estimation"]["use_odom"].as<bool>();

  RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"), "dt: %f", dt_);
  RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"), "use odom: %s",
              use_odom_ ? "true" : "false");
  for (const auto &name : foot_names) {
    RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"), "foot name: %s",
                name.c_str());//c_str()​​：将C++的std::string转换为C风格的字符串（const char*）
    cflag_.emplace_back(true);
  }

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data); //通信质量配置，此处设置为KeepLast(1)（仅保留最新消息）和传感器默认策略
  std::string topic_prefix =
      config_["global"]["topic_prefix"].as<std::string>();
  std::string imu_topic =
      config_["global"]["topic_names"]["imu"].as<std::string>();
  //订阅IMU传感器数据话题
  imu_subscription_ = nodeHandle_->create_subscription<sensor_msgs::msg::Imu>(
      topic_prefix + imu_topic, qos,
      std::bind(&StateEstimationLKF::imuCallback, this, std::placeholders::_1));

  std::string joints_topic =
      config_["global"]["topic_names"]["joints_state"].as<std::string>();
  joints_state_subscription_ =
      nodeHandle_->create_subscription<sensor_msgs::msg::JointState>(
          topic_prefix + joints_topic, qos,
          std::bind(&StateEstimationLKF::jointCallback, this,
                    std::placeholders::_1));

  std::string odom_topic =
      config_["global"]["topic_names"]["odom"].as<std::string>();
  odom_subscription_ =
      nodeHandle_->create_subscription<nav_msgs::msg::Odometry>(
          topic_prefix + odom_topic, qos,
          std::bind(&StateEstimationLKF::odomCallback, this,
                    std::placeholders::_1));
  //发布估计后的里程计信息（如位置、速度、姿态）
  odom_est_publisher_ = nodeHandle_->create_publisher<nav_msgs::msg::Odometry>(
      topic_prefix + odom_topic + "_est", qos);

  inner_loop_thread_ = std::thread(&StateEstimationLKF::innerLoop, this);
  run_.push(true);
  setup();
}

//安全终止内部线程，防止资源泄漏
StateEstimationLKF::~StateEstimationLKF() {
  run_.push(false);
  inner_loop_thread_.join();
}

//线性卡尔曼滤波（LKF）的初始化部分，主要用于设定状态估计器的初始状态和系统模型参数
void StateEstimationLKF::setup() {
  x_est.setZero();
  x_est.z() = 0.5;  // 设定初始状态的第3个分量（z方向）为0.5
  ps_.setZero();    // 速度状态初始化为0
  vs_.setZero();    // 加速度状态初始化为0

  //离散化A、B矩阵
  A_.setIdentity();                             // 初始化为单位矩阵
  A_.block<3, 3>(0, 3).diagonal().fill(dt_);    // 在[0:3, 3:6]块的对角线填充dt_

  B_.setZero();
  B_.block<3, 3>(3, 0).diagonal().fill(dt_);

  matrix_t C1 = matrix_t::Zero(3, 6);
  C1.leftCols(3).setIdentity();        // C1 = [I 0; 0 0]
  matrix_t C2 = matrix_t::Zero(3, 6);
  C2.rightCols(3).setIdentity();       // C2 = [0 0; I 0]

  C_.setZero();
  C_.block<3, 6>(0, 0) = C1;           // 观测前3维对应位置（直接测量）
  C_.block<3, 6>(3, 0) = C1;           // 观测中间3维再次对应位置（重复观测）
  C_.block<6, 6>(0, 6).diagonal().fill(-1.0);  // 对前6维施加-1的约束（先验知识）
  C_.block<3, 6>(6, 0) = C2;           // 观测第6-8维对应速度（可能来自另一传感器）
  C_.block<3, 6>(9, 0) = C2;           // 观测第9-11维再次对应速度（重复）
  C_(12, 8) = 1.0;                     // 第12维观测可能与第8维状态关联（如偏移补偿）
  C_(13, 11) = 1.0;                    // 第13维观测可能与第11维状态关联

  Sigma_.setZero();
  Sigma_.diagonal().fill(100.0);     // 初始状态协方差较大，表示低置信度

  Q0_.setIdentity();
  Q0_.topLeftCorner(3,3).diagonal().fill(dt_/20.0);     // 位置过程噪声较小
  Q0_.block<3,3>(3,3).diagonal().fill(dt_*9.81/20.0);   // 速度过程噪声较大（含重力影响）
  Q0_.bottomRightCorner(6,6).diagonal().fill(dt_);      // 其他状态噪声中等

  R0_.setIdentity();               // 观测噪声协方差初始化为单位矩阵
}

void StateEstimationLKF::setContactFlag(vector<bool> flag) { cflag_ = flag; }

void StateEstimationLKF::updateModeSchedule(
    std::shared_ptr<ModeSchedule> mode_schedule) {
  mode_schedule_buffer_.push(mode_schedule);
}

//角度运动估计
void StateEstimationLKF::angularMotionEstimate(
    const sensor_msgs::msg::Imu &imu_data, std::shared_ptr<vector_t> qpos,
    std::shared_ptr<vector_t> qvel) {
  
  // 将ROS消息中的四元数转换为Eigen库的Quaternion对象
  const auto &quat_imu = imu_data.orientation;
  Eigen::Quaternion<scalar_t> quat(quat_imu.w, quat_imu.x, quat_imu.y,
                                   quat_imu.z);
  // 更新机器人位姿（四元数部分）
  qpos->segment(3, 4) << quat_imu.x, quat_imu.y, quat_imu.z, quat_imu.w;

  // 更新机器人速度（角速度部分）
  qvel->segment(3, 3) << imu_data.angular_velocity.x, 
                        imu_data.angular_velocity.y, 
                        imu_data.angular_velocity.z;
}

//线性运动估计
void StateEstimationLKF::linearMotionEstimate(
    const sensor_msgs::msg::Imu &imu_data, std::shared_ptr<vector_t> qpos,
    std::shared_ptr<vector_t> qvel) {

  // 1. 将IMU四元数转换为Eigen库的Quaternion对象
  Eigen::Quaternion<scalar_t> quat(
      imu_data.orientation.w, imu_data.orientation.x, imu_data.orientation.y,
      imu_data.orientation.z);
  vector3_t b_acc(imu_data.linear_acceleration.x,
                  imu_data.linear_acceleration.y,
                  imu_data.linear_acceleration.z);
  // 2. 处理IMU线加速度：转换到全局坐标系并添加重力   注：w_acc意思是world_acceleration
  vector3_t w_acc = quat.toRotationMatrix() * b_acc;
  vector3_t g(0, 0, -9.81);
  w_acc = w_acc + g;

  // 3. 定义噪声协方差矩阵Q（过程噪声）和R（观测噪声）
  scalar_t noise_pimu = 0.02;  // 位置过程噪声标准差
  scalar_t noise_vimu = 0.02;  // 速度过程噪声标准差
  scalar_t noise_pfoot = 0.002;  // 足部压力位置噪声
  scalar_t noise_pimu_rel_foot = 0.001;  // 足部相对位置噪声
  scalar_t noise_vimu_rel_foot = 0.1;    // 足部相对速度噪声
  scalar_t noise_zfoot = 0.001;          // 足部z轴噪声

  matrix_t Q = matrix_t::Identity(12, 12);
  Q.topLeftCorner(3, 3) = Q0_.topLeftCorner(3, 3) * noise_pimu;
  Q.block<3, 3>(3, 3) = Q0_.block<3, 3>(3, 3) * noise_vimu;
  Q.block<6, 6>(6, 6) = Q0_.block<6, 6>(6, 6) * noise_pfoot;

  matrix_t R = matrix_t::Identity(14, 14);
  R.topLeftCorner(6, 6) = R0_.topLeftCorner(6, 6) * noise_pimu_rel_foot;
  R.block<6, 6>(6, 6) = R0_.block<6, 6>(6, 6) * noise_vimu_rel_foot;
  R.block<2, 2>(12, 12) = R0_.block<2, 2>(12, 12) * noise_zfoot;

  size_t q_idx = 0;
  size_t idx1 = 0;
  size_t idx2 = 0;

  vector4_t pzs = vector4_t::Zero();
  vector3_t p0, v0;
  p0 << x_est[0], x_est[1], x_est[2];
  v0 << x_est[3], x_est[4], x_est[5];

  // 4. 更新机器人状态和模式调度信息  
  pinocchioInterface_ptr->updateRobotState(*qpos, *qvel);
  auto mode_schedule = mode_schedule_buffer_.get();
  auto contact_flag = biped::modeNumber2StanceLeg(  // 获取足部接触标志
      mode_schedule->getModeFromPhase(0.0));         // 当前相位对应的步态模式
  auto ntlo = biped::getTimeOfNextLiftOff(0, mode_schedule);  // 下次抬脚时间
  const scalar_t gait_cycle = mode_schedule->gaitCycle();     // 步态周期长度

  // 5. 判断是否完全离地（基于垂直加速度）  注：w_acc意思是world_acceleration
  bool release_ = (abs(w_acc.z()) < 1.0);   // 若垂直加速度小于1.0，视为离地
      // 遍历所有足部，更新release_标志（所有足部均未触地时才为true）
  for (auto &flag : contact_flag) {
    release_ &= (!flag);
  }

  // 6. 遍历每个足部，计算动态噪声权重（信任度）  
  for (size_t i = 0; i < foot_names.size(); i++) {
    // 获取足部的位姿（位置和姿态）
    auto pose = pinocchioInterface_ptr->getFramePose(foot_names[i]);
    // 获取足部的6自由度速度（线速度+角速度）
    auto v = pinocchioInterface_ptr->getFrame6dVel_localWorldAligned(foot_names[i]);
    vector3_t p_f = pose.translation();  // 足部位置
    p_f.z() -= 0.02;  // 扣除足部半径（避免地面穿透）
    vector3_t v_f = v.linear();  // 足部线速度

    // 计算足部索引（状态向量中的位置）
    int i1 = 3 * i;      // 每个足部占3个状态（x,y,z）
    q_idx = 6 + i1;      // 位姿中足部的起始索引（前6维为全局位姿）
    idx1 = 6 + i1;       // 观测中足部的起始索引
    idx2 = 12 + i;       // 足部压力观测的索引

    // 初始化信任度（默认完全信任传感器）
    scalar_t trust = 1.0;
    // 如果足部触地或完全离地，动态调整信任度
    if (contact_flag[i] || release_) {
      const scalar_t trust_window = 0.15;  // 信任窗口阈值（15%阶段）
      scalar_t contact_phase = 1.0 - ntlo[i] / (0.5 * gait_cycle);  // 接触阶段比例
      // 根据接触阶段调整信任度（线性插值）
      if (contact_phase < trust_window) {
        trust = contact_phase / trust_window;  // 接触初期信任度递增
      } else if (contact_phase > (1.0 - trust_window)) {
        trust = (1.0 - contact_phase) / trust_window;  // 离地初期信任度递减
      }
    } else {
      trust = 0.0;  // 足部在空中时不信任其传感器数据
    }

    scalar_t high_suspect_number = 500.0;

    Q.block<3, 3>(q_idx, q_idx) = (1.0 + (1.0 - trust) * high_suspect_number) *
                                  Q.block<3, 3>(q_idx, q_idx);
    R.block<3, 3>(idx1, idx1) =
        (1.0 + (1.0 - trust) * high_suspect_number) * R.block<3, 3>(idx1, idx1);
    R(idx2, idx2) = (1.0 + (1.0 - trust) * high_suspect_number) * R(idx2, idx2);
    ps_.segment(i1, 3) = -p_f;
    pzs(i) = (1.0 - trust) * (p0(2) + p_f(2));
    vs_.segment(i1, 3) = (1.0 - trust) * v0 + trust * (-v_f);
  }

  // 构建观测向量y（位置、速度、足部压力）
  // y 是 14 维：ps_(6) + vs_(6) + pzs前2维 = 14
  // C_ 矩阵结构：y[0..5]=两足相对位置, y[6..11]=两足速度, y[12..13]=两足z坐标
  // pzs 声明为 vector4_t 但双足机器人只用前两个元素（foot0_z, foot1_z）
  Eigen::Matrix<scalar_t, 14, 1> y;
  y << ps_, vs_, pzs.head(2);

  // 卡尔曼滤波预测步骤
  vector_t x_pred = A_ * x_est + B_ * w_acc;  // x_pred = A*x_est + B*w_acc
  matrix_t Sigma_bar = A_ * Sigma_ * A_.transpose() + Q;  // 预测协方差

  // 卡尔曼增益计算
  matrix_t S = C_ * Sigma_bar * C_.transpose() + R;  // 卡尔曼增益分母
  vector_t correct = S.lu().solve(y - C_ * x_pred);  // K = S^{-1}*(y - C*x_pred)

  // 卡尔曼滤波校正步骤
  x_est = x_pred + Sigma_bar * C_.transpose() * correct;  // 更新状态估计

  // 更新协方差矩阵（对称化和截断）
  matrix_t SC_ = S.lu().solve(C_);  // 计算S^{-1}*C_
  Sigma_ = (matrix_t::Identity(12, 12) - Sigma_bar * C_.transpose() * SC_) 
           * Sigma_bar;            // 更新协方差
  Sigma_ = 0.5 * (Sigma_ + Sigma_.transpose());  // 强制对称（消除数值误差）

  // 防止协方差矩阵奇异（行列式检查）
  if (Sigma_.topLeftCorner(2, 2).determinant() > 1e-6) {
    Sigma_.topRightCorner(2, 10).setZero();  // 强制对角块稀疏化
    Sigma_.bottomLeftCorner(10, 2).setZero();
    Sigma_.topLeftCorner(2, 2) *= 0.1;  // 降低初始协方差权重
  }

  // 更新位姿和速度输出（转换到全局坐标系）
  qpos->head(3) = x_est.head(3);  // 更新位置（x,y,z）
  qvel->head(3) = quat.toRotationMatrix().transpose() * x_est.segment(3, 3);  // 转换速度到全局坐标系
}

void StateEstimationLKF::innerLoop() {
  rclcpp::Rate loop_rate(1.0 / dt_);

  const scalar_t t0 = nodeHandle_->now().seconds();
  while (rclcpp::ok() && run_.get()) {
    //确保所有必需的传感器数据已到达
    if (imu_msg_buffer.get().get() == nullptr ||
        mode_schedule_buffer_.get().get() == nullptr ||
        joint_state_msg_buffer.get().get() == nullptr ||
        (use_odom_ && odom_msg_buffer.get().get() == nullptr)) {
      continue;
    }
    // do not delete these three lines
    //获取发布的各种状态信息
    sensor_msgs::msg::Imu::SharedPtr imu_msg = imu_msg_buffer.get();
    sensor_msgs::msg::JointState::SharedPtr joint_state_msg =
        joint_state_msg_buffer.get();
    nav_msgs::msg::Odometry::SharedPtr odom_msg = odom_msg_buffer.get();

    const pin::Model &model_ = pinocchioInterface_ptr->getModel();
    std::shared_ptr<vector_t> qpos_ptr_ = std::make_shared<vector_t>(model_.nq);
    qpos_ptr_->setZero();
    std::shared_ptr<vector_t> qvel_ptr_ = std::make_shared<vector_t>(model_.nv);
    qvel_ptr_->setZero();

    for (size_t k = 0; k < joint_state_msg->name.size(); k++) {
      if (static_cast<int>(k) < model_.njoints &&
          model_.existJointName(joint_state_msg->name[k])) {
        pin::Index id = model_.getJointId(joint_state_msg->name[k]) - 2;
        (*qpos_ptr_)[id + 7] = joint_state_msg->position[k];
        (*qvel_ptr_)[id + 6] = joint_state_msg->velocity[k];
      }
    }

    if (use_odom_) {
      // 从里程计消息提取数据并转换到机器人基座坐标系
      const auto &orientation = odom_msg->pose.pose.orientation;
      const auto &position = odom_msg->pose.pose.position;
      const auto &vel = odom_msg->twist.twist.linear;
      const auto &ang_vel = odom_msg->twist.twist.angular;

      Eigen::Quaternion<scalar_t> quat(orientation.w, orientation.x,
                                       orientation.y, orientation.z);
      matrix3_t rot = quat.toRotationMatrix();
      (*qpos_ptr_).head(3) << position.x, position.y, position.z;
      (*qvel_ptr_).head(3) << vel.x, vel.y, vel.z;
      (*qvel_ptr_).head(3) = rot.transpose() * (*qvel_ptr_).head(3);

      (*qpos_ptr_).segment(3, 4) << orientation.x, orientation.y, orientation.z,
          orientation.w;
      (*qvel_ptr_).segment(3, 3) << ang_vel.x, ang_vel.y, ang_vel.z;
    } else {
      // const auto touch_sensor_data = touch_msg_buffer.get()->value;
      // for (auto &data : touch_sensor_data) {
      //   RCLCPP_INFO(rclcpp::get_logger("StateEstimationLKF"),
      //   "touch_sensor_data: %f", data);
      // }

      // 通过 IMU 和卡尔曼滤波估计状态
      angularMotionEstimate(*imu_msg, qpos_ptr_, qvel_ptr_);
      linearMotionEstimate(*imu_msg, qpos_ptr_, qvel_ptr_);
    }

    if (nodeHandle_->now().seconds() - t0 > 0.3) {
      qpos_ptr_buffer.push(qpos_ptr_);
      qvel_ptr_buffer.push(qvel_ptr_);
    }
    // 构建并发布里程计消息
    auto message = nav_msgs::msg::Odometry();
    {
      message.header.frame_id = robot_name;
      message.header.stamp = nodeHandle_->now();
      message.pose.pose.position.x = qpos_ptr_->x();
      message.pose.pose.position.y = qpos_ptr_->y();
      message.pose.pose.position.z = qpos_ptr_->z();
      message.pose.pose.orientation.w = (*qpos_ptr_)[6];
      message.pose.pose.orientation.x = (*qpos_ptr_)[3];
      message.pose.pose.orientation.y = (*qpos_ptr_)[4];
      message.pose.pose.orientation.z = (*qpos_ptr_)[5];

      Eigen::Quaternion<scalar_t> quat((*qpos_ptr_)[6], (*qpos_ptr_)[3],
                                       (*qpos_ptr_)[4], (*qpos_ptr_)[5]);
      vector3_t vel_world = quat.toRotationMatrix() * qvel_ptr_->head(3);
      message.twist.twist.linear.x = vel_world.x();
      message.twist.twist.linear.y = vel_world.y();
      message.twist.twist.linear.z = vel_world.z();
      message.twist.twist.angular.x = (*qvel_ptr_)[3];
      message.twist.twist.angular.y = (*qvel_ptr_)[4];
      message.twist.twist.angular.z = (*qvel_ptr_)[5];
    }
    odom_est_publisher_->publish(message);

    /* RCLCPP_INFO_STREAM(rclcpp::get_logger("StateEstimationLKF"),
                       "qpos: " << (*qpos_ptr_).transpose() << "\n");
    RCLCPP_INFO_STREAM(rclcpp::get_logger("StateEstimationLKF"),
                       "qvel: " << (*qvel_ptr_).transpose() << "\n"); */

    loop_rate.sleep();
  }
}

std::shared_ptr<vector_t> StateEstimationLKF::getQpos() {
  return qpos_ptr_buffer.get();
}

std::shared_ptr<vector_t> StateEstimationLKF::getQvel() {
  return qvel_ptr_buffer.get();
}

void StateEstimationLKF::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name) {
    imu_msg_buffer.push(msg);
  }
}

void StateEstimationLKF::jointCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name) {
    joint_state_msg_buffer.push(msg);
  }
}

void StateEstimationLKF::setImuMsg(sensor_msgs::msg::Imu::SharedPtr msg) {
  if (msg->header.frame_id == robot_name) {
    imu_msg_buffer.push(msg);
  }
}

void StateEstimationLKF::setJointsMsg(
    sensor_msgs::msg::JointState::SharedPtr msg) {
  if (msg->header.frame_id == robot_name) {
    joint_state_msg_buffer.push(msg);
  }
}

void StateEstimationLKF::odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) const {
  if (msg->header.frame_id == robot_name) {
    odom_msg_buffer.push(msg);
  }
}

}  // namespace clear