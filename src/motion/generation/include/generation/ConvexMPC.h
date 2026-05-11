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

  void collectTrainingData(const vector_t& x_now);

  void writeTrainingData();

  void collectWindowData(const vector_t& x_now, const vector_t& delta0);

  void writeWindowData();

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
  std::unique_ptr<clear::MLPNetwork> mlp_network_;       // v2 模型（或单模型模式下的唯一模型）
  std::unique_ptr<clear::MLPNetwork> mlp_network_v1_;    // 混合模式：前 N 步用 v1（LKF输入分布）
  int mlp_v1_steps_ = 1;                                // v1 负责的步数（默认1，即仅 k=0）
  bool use_mlp_residual_ = false;
  bool mlp_full_recursive_ = false;  // 全递归 MLP（每步调用），false 时为 Method C'（只修正 b_0）
  int mlp_recursive_steps_ = -1;    // 全递归时生效：MLP 修正的步数，-1 = 全部 N 步
  double mlp_residual_clamp_ = 0.5;

  // B3 AL-full (codex collapse-review verdict V2): augmented-Lagrangian
  // friction-constrained PMP update. Default false → legacy path unchanged.
  bool use_augmented_lagrangian_ = false;
  double al_rho_ = 1.0e-2;
  int al_outer_iters_ = 1;
  bool al_final_projection_safety_ = true;
  double al_active_threshold_ = 0.0;
  // B3 Fix A (codex full-review V2): persistent AL multipliers across MPC
  // ticks. Without this, every tick rebuilds the optimizer and ν is reset to
  // 0, degrading the update to a one-shot penalty rather than a true AL
  // outer iteration. Owned here, handed to PMPOptimizer per tick.
  std::vector<vector_t> al_nu_upper_state_;
  std::vector<vector_t> al_nu_lower_state_;

  // 训练数据采集
  bool collect_training_data_ = false;
  int max_training_samples_ = 10000;   // 最多采集样本数
  vector_t prev_x_;                   // 上一步的状态 x_k
  vector_t prev_u_;                   // 上一步的控制 u_k
  vector_t prev_x_next_est_;          // 上一步的名义预测 x_next_est
  vector_t prev_delta_mlp_;           // 上一步的 MLP 影子预测（不施加，仅记录）
  bool has_prev_data_ = false;
  std::vector<std::string> training_rows_;
  bool training_data_written_ = false;

  // 窗口数据采集
  bool collect_window_data_ = false;
  int max_window_samples_ = 500;
  int window_count_ = 0;
  bool window_data_written_ = false;
  std::vector<std::string> window_rows_;

  // 上一窗口缓存（等待 k=0 的 x_next_true）
  struct WindowStep {
    vector_t x_pred, u, x_next_est, x_next_true;
  };
  std::vector<WindowStep> prev_window_buf_;
  bool has_prev_window_ = false;
};

} // namespace clear
