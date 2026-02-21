#pragma once
#include <core/gait/ModeSchedule.h>
#include <core/gait/MotionPhaseDefinition.h>
#include <core/trajectory/ReferenceBuffer.h>

#include <core/misc/Buffer.h>
#include <pinocchio/Orientation.h>
#include <pinocchio/PinocchioInterface.h>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <memory>
#include "generation/PMPOptimizer.h"
#include "estimation/MLPNetwork.h"

using namespace rclcpp;

namespace clear {
class ConvexMPC {
public:
  ConvexMPC(Node::SharedPtr nodeHandle,
            std::shared_ptr<PinocchioInterface> pinocchioInterface_ptr,
            std::shared_ptr<ReferenceBuffer> referenceBuffer);

  ~ConvexMPC();

  void optimize();

  void setVelCmd(vector3_t vd, scalar_t yawd);

  void setHeightCmd(scalar_t h);

  void generateTrajRef();

private:

  void getDynamics(scalar_t time_cur, size_t k,
                   const std::shared_ptr<ModeSchedule> mode_schedule);

  void
  getInequalityConstraints(size_t k, size_t N,
                           const std::shared_ptr<ModeSchedule> mode_schedule);

  void getCosts(scalar_t time_cur, size_t k, size_t N,
                const std::shared_ptr<ModeSchedule> mode_schedule);

  void fitTraj(scalar_t time_cur, size_t N);

  void evaluateAndExportMetrics(double t0, double dt, double solve_time_ms);

private:
  Node::SharedPtr nodeHandle_;
  std::shared_ptr<PinocchioInterface> pinocchioInterface_ptr_;
  std::shared_ptr<ReferenceBuffer> referenceBuffer_;

  std::string base_name;
  std::vector<std::string> foot_names;

  scalar_t h_des = 0.58;
  
  scalar_t dt_ = 0.02;
  const scalar_t grav_ = 9.81;
  scalar_t total_mass_;
  matrix3_t Ig_;
  const scalar_t mu_ = 0.5; //0.5
  matrix_t weight_;
  vector3_t rpy_des_start;
  bool first_run_ = true;

  std::vector<clear::OcpStage> ocp_;
  std::vector<clear::OcpSolution> solution_;

  vector3_t vel_cmd;
  scalar_t yawd_;

  vector3_t rpy_start;
  vector3_t pos_start;
  bool first_run = true;
  scalar_t t0 = 0.0;

  // 添加一段程序用于计时评估指标
  double time_prog_begin;

  // MLP 残差补偿
  std::unique_ptr<clear::MLPNetwork> mlp_network_;
  bool use_mlp_residual_ = false;
  double mlp_residual_clamp_ = 0.5;
};

} // namespace clear
