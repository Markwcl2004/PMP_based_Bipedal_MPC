#include "generation/FootholdOptimization.h"

#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

/*
该类是双足机器人步态生成的核心模块，负责 ​​动态计算足部目标接触点（footholds）​​，主要解决以下问题：

​​运动学适配​​：根据机器人当前姿态与步态周期，预测足部未来接触点。
​​动力学补偿​​：补偿基座速度偏差、垂直运动与航向角变化的影响。
​​安全约束​​：限制足部移动范围，避免步态失效
*/

namespace clear {

FootholdOptimization::FootholdOptimization(
    Node::SharedPtr nodeHandle,
    std::shared_ptr<PinocchioInterface> pinocchioInterface_ptr,
    std::shared_ptr<ReferenceBuffer> referenceBuffer)
    : nodeHandle_(nodeHandle),
      pinocchioInterface_ptr_(pinocchioInterface_ptr),
      referenceBuffer_(referenceBuffer) {
  // 从参数服务器加载配置文件路径
  const std::string config_file_ = nodeHandle_->get_parameter("/config_file")
                                       .get_parameter_value()
                                       .get<std::string>();
  // 解析YAML配置文件
  auto config_ = YAML::LoadFile(config_file_);

  // 读取足部名称、基座名称
  foot_names = config_["model"]["foot_names"].as<std::vector<std::string>>();
  base_name = config_["model"]["base_name"].as<std::string>();

  vector_t qpos, qvel;
  //nq() 和 nv()​​：分别返回机器人模型的关节数（自由度数）和速度维度
  //将机器人所有关节的位置和速度设置为零（即默认静止状态）
  qpos.setZero(pinocchioInterface_ptr_->nq());
  qvel.setZero(pinocchioInterface_ptr_->nv());
  
  //Pinocchio接口的关键方法，用于根据当前的关节状态（qpos, qvel）更新机器人模型
  pinocchioInterface_ptr_->updateRobotState(qpos, qvel);
  /*
    名义位置是足部（或末端执行器）在设计阶段定义的 ​​理想接触点​​ 或 ​​默认位置​​，通常基于机器人模型
    （如 URDF/SDF 文件）的几何参数计算得出。
    ​​仿真与实际的桥梁​​：在仿真环境中，名义位置可能与实际物理机器人的足部位置存在差异（例如传感器误差或模型简化），因此需要通过标定或调整来对齐。
  */
  for (const auto &foot : foot_names) {
    footholds_nominal_pos[foot] =
        pinocchioInterface_ptr_->getFramePose(foot).translation();
    footholds_nominal_pos[foot].z() = 0.0;
    footholds_nominal_pos[foot].y() *= 0.8;
    std::cout << foot << ": " << footholds_nominal_pos[foot].transpose()
              << "\n";
  }

  nf = foot_names.size();
}

FootholdOptimization::~FootholdOptimization() {}

void FootholdOptimization::optimize() {
  //清空现有足部目标位置，准备重新计算
  footholds_ = std::map<std::string, std::pair<scalar_t, vector3_t>>();
  auto base_pos_ref_traj = referenceBuffer_->getIntegratedBasePosTraj(); //基座的全局位置轨迹（如三次样条曲线）
  auto base_rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();  //基座的滚转-俯仰-偏航（RPY）角轨迹
  auto mode_schedule = referenceBuffer_->getModeSchedule();
  if (mode_schedule.get() == nullptr || base_pos_ref_traj.get() == nullptr ||
      base_rpy_traj.get() == nullptr) {
    referenceBuffer_->setFootholds(footholds_);
  }
  heuristic();
  referenceBuffer_->setFootholds(footholds_);
}

// 该函数通过 ​​运动学补偿​​ 和 ​​动力学耦合​​，计算双足机器人足部的期望接触点（footholds_）
void FootholdOptimization::heuristic() {
  //获取当前时间、基座参考轨迹、步态模式、机器人位姿/速度
  const scalar_t t = nodeHandle_->now().seconds();

  auto base_pos_ref_traj = referenceBuffer_->getIntegratedBasePosTraj();
  auto base_rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();
  auto mode_schedule = referenceBuffer_->getModeSchedule();
  const auto base_pose = pinocchioInterface_ptr_->getFramePose(base_name);
  const auto base_twist =
      pinocchioInterface_ptr_->getFrame6dVel_localWorldAligned(base_name);
  const scalar_t gait_cycle = mode_schedule->duration();
  auto contact_flag =
      biped::modeNumber2StanceLeg(mode_schedule->getModeFromPhase(0.0));

  //  计算期望速度与角速度
  vector3_t v_des = base_pos_ref_traj->derivative(t, 1);
  vector3_t rpy_dot_des = base_rpy_traj->derivative(t, 1);

  //  根据步态模式表，计算某个足部（如左足）下次触地的时间
  auto ntc = biped::getTimeOfNextTouchDown(0, mode_schedule);

  const scalar_t p_rel_x_max = 0.5f;
  const scalar_t p_rel_y_max = 0.3f;

  for (size_t i = 0; i < nf; i++) {
    const auto &foot_name = foot_names[i];
    // swingTimeRemain：若足部处于支撑相则为 0，否则为剩余摆动时间（ntc[i] - t）
    /*
        支撑相足部无需移动（swingTimeRemain = 0）。
        摆动相足部需计算到达目标位置的时间（ntc[i] - t）
    */
    const scalar_t swingTimeRemain = contact_flag[i] ? 0.0 : ntc[i];
    /*
        预测未来姿态​​：在 t + gait_cycle/2.0 时刻获取基座的旋转矩阵（航向角预测）
        将足部的预设位置（footholds_nominal_pos）转换到全局坐标系
        p_YawCorrected=R(θ)⋅p_nom
    */
    vector3_t pYawCorrected =
        toRotationMatrix(base_rpy_traj->evaluate(t + gait_cycle / 2.0)) *
        footholds_nominal_pos[foot_name];

    scalar_t pfx_rel, pfy_rel;
    std::pair<scalar_t, vector3_t> foothold;
    /*
      first：时间戳（秒），表示该足部目标点的生效时间。
      second：三维位置向量 [x, y, z]，表示足部在基座坐标系下的目标位姿。
    */
    foothold.first = ntc[i] + t;
    foothold.second = base_pose.translation() +
                      (pYawCorrected + std::max(0.0, swingTimeRemain) * v_des);

    // 计算相对偏移
    pfx_rel = 0.5 * gait_cycle * v_des.x() +    // 基于​匀速运动假设​​，预测基座在未来半个步态周期（0.5 * gait_cycle）内的位移
              0.4 * (base_twist.linear().x() - v_des.x()) + // 补偿基座 ​​实际线速度​​ 与 ​​期望线速度​​ 的差异，提升跟踪精度
              (0.5 * base_pose.translation().z() / 9.81) *  // 补偿因 ​​基座垂直运动（Z轴）​​ 与 ​​偏航角速度（ψ）​​ 耦合产生的侧向位移
                  (base_twist.linear().y() * rpy_dot_des.z());
    pfy_rel = 0.5 * gait_cycle * v_des.y() +
              0.08 * (base_twist.linear().y() - v_des.y()) +
              (0.5 * base_pose.translation().z() / 9.81) *
                  (-base_twist.linear().x() * rpy_dot_des.z());

    // 限幅处理
    pfx_rel = std::min(std::max(pfx_rel, -p_rel_x_max), p_rel_x_max);
    pfy_rel = std::min(std::max(pfy_rel, -p_rel_y_max), p_rel_y_max);

    
    foothold.second.x() += pfx_rel;
    foothold.second.y() += pfy_rel;
    foothold.second.z() = 0.03;  // 考虑足部半径
    footholds_[foot_name] = foothold;
  }
}

}  // namespace clear