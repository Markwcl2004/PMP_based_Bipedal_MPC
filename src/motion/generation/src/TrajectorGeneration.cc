#include "generation/TrajectorGeneration.h"

#include <core/misc/Benchmark.h>
#include <core/misc/NumericTraits.h>
#include <pinocchio/Orientation.h>
#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

/*
    该类是机器人运动轨迹生成的核心模块，负责 ​​基座轨迹、足部轨迹的生成与协调​​
*/

namespace clear {

TrajectorGeneration::TrajectorGeneration(Node::SharedPtr nodeHandle)
    : nodeHandle_(nodeHandle) {
  /*
      ​​配置加载​​：从ROS参数服务器读取URDF模型路径、足部名称、基座名称和生成频率。
      ​​运动学模型初始化​​：通过Pinocchio接口加载URDF模型，构建机器人刚体树。
      ​​优化器实例化​​：创建足部优化器（FootholdOptimization）和基座轨迹优化器（ConvexMPC）。
      ​​线程启动​​：调用innerLoop()开始周期性轨迹生成
  */
  const std::string config_file_ = nodeHandle_->get_parameter("/config_file")
                                       .get_parameter_value()
                                       .get<std::string>();

  auto config_ = YAML::LoadFile(config_file_);

  std::string model_package = config_["model"]["package"].as<std::string>();
  std::string urdf =
      ament_index_cpp::get_package_share_directory(model_package) +
      config_["model"]["urdf"].as<std::string>();
  RCLCPP_INFO(rclcpp::get_logger("TrajectorGeneration"), "model file: %s",
              urdf.c_str());
  pinocchioInterface_ptr_ = std::make_shared<PinocchioInterface>(urdf.c_str());

  foot_names = config_["model"]["foot_names"].as<std::vector<std::string>>();

  base_name = config_["model"]["base_name"].as<std::string>();
  RCLCPP_INFO(rclcpp::get_logger("TrajectorGeneration"),
              "[TrajectorGeneration]  base name: %s", base_name.c_str());

  freq_ = config_["generation"]["frequency"].as<scalar_t>();
  RCLCPP_INFO(rclcpp::get_logger("TrajectorGeneration"),
              "[TrajectorGeneration] frequency: %f", freq_);

  contact_flag_ = {true, true};
  
  /*
    Allocates memory for an object and initialize the object with the supplied arguments. 
    Returns a std::shared_ptr object managing the newly created object.  
  */
  referenceBuffer_ = std::make_shared<ReferenceBuffer>();
  
  // 足部优化器
  footholdOpt_ptr = std::make_shared<FootholdOptimization>(
      nodeHandle_, pinocchioInterface_ptr_, referenceBuffer_);

  // 基座优化器
  baseOpt_ptr = std::make_shared<ConvexMPC>(
      nodeHandle_, pinocchioInterface_ptr_, referenceBuffer_);

  run_.push(true);

  inner_loop_thread_ = std::thread(&TrajectorGeneration::innerLoop, this);
}

TrajectorGeneration::~TrajectorGeneration() {
  run_.push(false);
  inner_loop_thread_.join();
}

void TrajectorGeneration::setHeightCmd(scalar_t h) {
  // 向基座优化器传递高度指令，调整基座轨迹的全局高度
  baseOpt_ptr->setHeightCmd(h);
}

void TrajectorGeneration::updateCurrentState(
    std::shared_ptr<vector_t> qpos_ptr, std::shared_ptr<vector_t> qvel_ptr) {
  // 更新关节位置和速度缓冲区，供后续轨迹生成使用
  qpos_ptr_buffer.push(qpos_ptr);
  qvel_ptr_buffer.push(qvel_ptr);
}

void TrajectorGeneration::updateModeSchedule(
    std::shared_ptr<ModeSchedule> mode_schedule) {
  referenceBuffer_->setModeSchedule(mode_schedule);
}

std::shared_ptr<ReferenceBuffer> TrajectorGeneration::getReferenceBuffer() {
  return referenceBuffer_;
}

void TrajectorGeneration::innerLoop() {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  benchmark::RepeatedTimer timer_;
  rclcpp::Rate loop_rate(freq_);

  RCLCPP_INFO(rclcpp::get_logger("TrajectorGeneration"),
              "start generation loop");

  while (rclcpp::ok() && run_.get()) {
    timer_.startTimer();
    if (qpos_ptr_buffer.get().get() == nullptr ||
        qvel_ptr_buffer.get().get() == nullptr ||
        referenceBuffer_->getModeSchedule() == nullptr) {
      continue;
    } else {
      std::shared_ptr<vector_t> qpos_ptr = qpos_ptr_buffer.get();
      std::shared_ptr<vector_t> qvel_ptr = qvel_ptr_buffer.get();
      pinocchioInterface_ptr_->updateRobotState(*qpos_ptr, *qvel_ptr);

      baseOpt_ptr->generateTrajRef();

      footholdOpt_ptr->optimize();

      generateFootTraj();

      baseOpt_ptr->optimize();
    }
    timer_.endTimer();
    loop_rate.sleep();
  }
  RCLCPP_INFO(rclcpp::get_logger("TrajectorGeneration"),
              "[TrajectorGeneration] max time %f ms,  average time %f ms",
              timer_.getMaxIntervalInMilliseconds(),
              timer_.getAverageInMilliseconds());
}

/*
为每个足部生成三次样条轨迹，确保平滑过渡

1初始化轨迹端点​​：若首次运行，记录当前位姿为起点；根据步态周期预测终点。
​2​动态调整端点​​：根据接触状态变化（contact_flag）更新轨迹起点或终点。
​3​三次样条插值​​：
  输入：时间-位置序列（包含起点、中间点、终点）。
  输出：分段三次多项式轨迹。
​4​边界条件​​：
  初速度与末速度为零（BoundaryType::first_deriv设为vector3_t::Zero()）。
*/
void TrajectorGeneration::generateFootTraj() {
  auto footholds = referenceBuffer_->getFootholds();
  const scalar_t t_now = nodeHandle_->now().seconds();
  auto mode_schedule = referenceBuffer_->getModeSchedule();
  if (xf_start_.empty()) {
    for (size_t k = 0; k < foot_names.size(); k++) {
      const auto &foot_name = foot_names[k];
      vector3_t pos =
          pinocchioInterface_ptr_->getFramePose(foot_name).translation();
      std::pair<scalar_t, vector3_t> xs;
      xs.first = nodeHandle_->now().seconds();    // 当前时间戳
      xs.second = pos;    // 当前足部位置
      xf_start_[foot_name] = std::move(xs);   //  std::move 并不实际"移动"对象，而是通过类型转换告诉编译器："这个对象可以被安全地移动"
    }
  }
  if (xf_end_.empty()) {
    for (size_t k = 0; k < foot_names.size(); k++) {
      const auto &foot_name = foot_names[k];
      vector3_t pos =
          pinocchioInterface_ptr_->getFramePose(foot_name).translation();
      std::pair<scalar_t, vector3_t> xe;
      xe.first = t_now + mode_schedule.get()->duration();   // 当前时间 + 步态周期时长
      xe.second = pos;    // 初始终点位置
      xf_end_[foot_name] = std::move(xe);
    }
  }
  // 遍历足部并动态调整端点​
  std::map<std::string, std::shared_ptr<CubicSplineTrajectory>> foot_pos_traj;

  // 获得足部接触状态
  auto contact_flag =
      biped::modeNumber2StanceLeg(mode_schedule->getModeFromPhase(0.0));
  for (size_t k = 0; k < foot_names.size(); k++) {
    const auto &foot_name = foot_names[k];
    vector3_t pos =
        pinocchioInterface_ptr_->getFramePose(foot_name).translation();
    // 如果足部接触状态发生变化
    if (contact_flag_[k] != contact_flag[k]) {
      std::pair<scalar_t, vector3_t> xs;
      /*
          若 t_now 接近 0（如初始化状态），直接减去 epsilon 可能导致负值。
          limitEpsilon 返回一个合理的下界（如 1e-5），确保时间戳始终为正
      */
      xs.first = t_now - numeric_traits::limitEpsilon<scalar_t>();
      xs.second = pos;
      xf_start_[foot_name] = std::move(xs);
    }

    // 根据当前接触状态调整终点
    if (contact_flag[k]) {  // 支撑相
      if (contact_flag_[k] != contact_flag[k]) {
        std::pair<scalar_t, vector3_t> xe;
        xe.first = t_now + mode_schedule->timeLeftInMode(0.0);  // 剩余时间
        xe.second = pos;
        xf_end_[foot_name] = std::move(xe);
      }
    } else {  // 摆动相
      std::pair<scalar_t, vector3_t> xe;
      xe.first = footholds[foot_name].first;  // 使用预定义的名义位置时间
      xe.second = footholds[foot_name].second;
      xf_end_[foot_name] = std::move(xe);
    }

    // 确保时间跨度合理（避免时间倒流）
    if (xf_start_[foot_name].first <
        xf_end_[foot_name].first - mode_schedule.get()->duration()) {
      xf_start_[foot_name].first =
          xf_end_[foot_name].first - mode_schedule.get()->duration();
      xf_start_[foot_name].second = pos;
    }

    // 物理约束：限制 Z 轴高度。足部轨迹的 Z 轴高度不能超过基座高度减去 0.2 米，防止足部穿透地面
    vector_t base_pos =
        pinocchioInterface_ptr_->getFramePose(base_name).translation();
    xf_start_[foot_name].second.z() =
        std::min(xf_start_[foot_name].second.z(), base_pos.z() - 0.2);
    xf_end_[foot_name].second.z() =
        std::min(xf_end_[foot_name].second.z(), base_pos.z() - 0.2);

    
    /*
    构建时间-位置序列

    ​​时间点​​：包含起点、中间点（时间中点）、终点。
    ​​位置点​​：
        起点和终点为实际位置（xf_start_ 和 xf_end_）。
        中间点为起点和终点的几何中点，若为摆动相则抬高 0.1 米（避免拖地）。
    */
    std::vector<scalar_t> time;
    std::vector<vector_t> pos_t;
    time.push_back(xf_start_[foot_name].first);     // 起点时间
    pos_t.push_back(xf_start_[foot_name].second);   // 起点位置
    time.push_back(0.5 *
                   (xf_start_[foot_name].first + xf_end_[foot_name].first));  // 中间时间点
    vector3_t middle_pos =
        0.5 * (xf_start_[foot_name].second + xf_end_[foot_name].second);
    middle_pos.z() += contact_flag[k] ? 0.0 : 0.1;
    pos_t.push_back(middle_pos);
    time.push_back(xf_end_[foot_name].first);     // 终点时间
    pos_t.push_back(xf_end_[foot_name].second);   // 终点位置

    /* std::cout << "############# " << foot_name << ": " << pos.transpose()
              << " ##############\n";
    for (size_t i = 0; i < time.size(); i++) {
      std::cout << " t: " << time[i] - time.front()
                << " pos: " << pos_t[i].transpose() << "\n";
    } */

    // 三次样条插值​
    auto cubicspline_ = std::make_shared<CubicSplineTrajectory>(
        3, CubicSplineInterpolation::SplineType::cspline);
    cubicspline_->set_boundary(
        CubicSplineInterpolation::BoundaryType::first_deriv, vector3_t::Zero(),
        CubicSplineInterpolation::BoundaryType::first_deriv, vector3_t::Zero());
    cubicspline_->fit(time, pos_t);
    /* std::cout << "############# opt " << foot_name << " ##############\n";
    for (size_t i = 0; i < time.size(); i++) {
      std::cout
          << " t: " << time[i] - time.front() << " pos: "
          << foot_traj_ptr->pos_trajectoryPtr->evaluate(time[i]).transpose()
          << "\n";
    } */
    foot_pos_traj[foot_name] = std::move(cubicspline_);
  }
  contact_flag_ = contact_flag;   // 同步最新接触状态
  referenceBuffer_->setFootPosTraj(foot_pos_traj);    // 保存轨迹至参考缓冲区
}

void TrajectorGeneration::setVelCmd(vector3_t vd, scalar_t yawd) {
  baseOpt_ptr->setVelCmd(vd, yawd);
}

}  // namespace clear
