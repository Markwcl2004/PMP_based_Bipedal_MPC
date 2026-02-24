#include "generation/ConvexMPC.h"
#include <pinocchio/Orientation.h>
#include <rcpputils/asserts.hpp>
#include <yaml-cpp/yaml.h>
#include "generation/PMPOptimizer.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace clear {

// std::vector<OcpStage> pmp_ocp_;
// std::vector<OcpSolution> pmp_solution_;

ConvexMPC::ConvexMPC(Node::SharedPtr nodeHandle,
                     std::shared_ptr<PinocchioInterface> pinocchioInterface_ptr,
                     std::shared_ptr<ReferenceBuffer> referenceBuffer)
    : nodeHandle_(nodeHandle), pinocchioInterface_ptr_(pinocchioInterface_ptr),
      referenceBuffer_(referenceBuffer) {

  const std::string config_file_ = nodeHandle_->get_parameter("/config_file")
                                       .get_parameter_value()
                                       .get<std::string>();

  auto config_ = YAML::LoadFile(config_file_);
  foot_names = config_["model"]["foot_names"].as<std::vector<std::string>>();
  base_name = config_["model"]["base_name"].as<std::string>();
  scalar_t freq_ = config_["generation"]["frequency"].as<scalar_t>();
  dt_ = 0.02;   // 控制周期 20ms（与 My_MPC 一致）

  // 总质量与权重矩阵（状态误差和控制输入的权重），定义状态误差（位置、速度、姿态）与控制输入的惩罚权重
  total_mass_ = pinocchioInterface_ptr_->total_mass();
  weight_.setZero(12, 12);
  // 权重与 My_MPC 一致（dt=0.02 调参版本）
  weight_.diagonal() << 10, 10, 40, 0.2, 0.2, 0.1, 4, 4, 4, 0.1, 0.1, 0.1;

  // HPIPM求解器参数设置（凸优化配置）  平衡求解速度与精度
  // solver_settings.mode = hpipm::HpipmMode::Speed;
  // solver_settings.iter_max = 30;
  // solver_settings.alpha_min = 1e-8;
  // solver_settings.mu0 = 1e2;
  // solver_settings.tol_stat = 1e-04;
  // solver_settings.tol_eq = 1e-04;
  // solver_settings.tol_ineq = 1e-04;
  // solver_settings.tol_comp = 1e-04;
  // solver_settings.reg_prim = 1e-12;
  // solver_settings.pred_corr = 1;
  // solver_settings.ric_alg = 0;
  // solver_settings.split_step = 1;

  vel_cmd.setZero();
  yawd_ = 0.0;
  t0 = nodeHandle->now().seconds();
  time_prog_begin = nodeHandle->now().seconds();

  // MLP 初始化：加载与控制启用分离
  // use_mlp_residual_ 控制是否施加到控制，mlp_network_ 只要提供路径就加载（用于影子记录）
  if (config_["generation"]["use_mlp_residual"]) {
    use_mlp_residual_ = config_["generation"]["use_mlp_residual"].as<bool>(false);
  }
  if (config_["generation"]["mlp_residual_clamp"]) {
    mlp_residual_clamp_ = config_["generation"]["mlp_residual_clamp"].as<double>(0.5);
  }
  if (config_["generation"]["mlp_full_recursive"]) {
    mlp_full_recursive_ = config_["generation"]["mlp_full_recursive"].as<bool>(false);
  }
  if (config_["generation"]["mlp_recursive_steps"]) {
    mlp_recursive_steps_ = config_["generation"]["mlp_recursive_steps"].as<int>(-1);
  }
  if (config_["generation"]["mlp_config"]) {
    std::string mlp_config_path = config_["generation"]["mlp_config"].as<std::string>("");
    if (!mlp_config_path.empty()) {
      try {
        mlp_network_ = std::make_unique<MLPNetwork>(mlp_config_path);
        RCLCPP_INFO(nodeHandle_->get_logger(),
                    "[ConvexMPC] MLP loaded (input=%d, output=%d, clamp=%.2f) — %s",
                    mlp_network_->getInputDim(), mlp_network_->getOutputDim(),
                    mlp_residual_clamp_,
                    use_mlp_residual_ ? (mlp_full_recursive_ ? (mlp_recursive_steps_ > 0 ? "RECURSIVE (limited steps)" : "FULL RECURSIVE (every step)") : "Method C' (b_0 only)") : "shadow mode only");
        if (use_mlp_residual_ && mlp_full_recursive_ && mlp_recursive_steps_ > 0) {
          RCLCPP_INFO(nodeHandle_->get_logger(),
                      "[ConvexMPC] MLP recursive steps: %d / N", mlp_recursive_steps_);
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN(nodeHandle_->get_logger(),
                    "[ConvexMPC] MLP load failed: %s", e.what());
        use_mlp_residual_ = false;
        mlp_network_.reset();
      }
    }
  }
  if (use_mlp_residual_ && !mlp_network_) {
    RCLCPP_WARN(nodeHandle_->get_logger(),
                "[ConvexMPC] use_mlp_residual=true but MLP not loaded, disabling");
    use_mlp_residual_ = false;
  }
  if (!mlp_network_) {
    RCLCPP_INFO(nodeHandle_->get_logger(), "[ConvexMPC] No MLP loaded");
  }

  // 训练数据采集开关
  if (config_["generation"]["collect_training_data"]) {
    collect_training_data_ = config_["generation"]["collect_training_data"].as<bool>(false);
  }
  if (config_["generation"]["max_training_samples"]) {
    max_training_samples_ = config_["generation"]["max_training_samples"].as<int>(10000);
  }
  if (collect_training_data_) {
    RCLCPP_INFO(nodeHandle_->get_logger(),
                "[ConvexMPC] Training data collection ENABLED (max %d samples -> mpc_data/)",
                max_training_samples_);
  }

  RCLCPP_INFO(rclcpp::get_logger("ConvexMPC"), "ConvexMPC: Construction done");
}

ConvexMPC::~ConvexMPC() {}

void ConvexMPC::setVelCmd(vector3_t vd, scalar_t yawd) {
  vel_cmd = vd;
  yawd_ = yawd;
}

void ConvexMPC::setHeightCmd(scalar_t h) { h_des = h; }

void ConvexMPC::generateTrajRef() {
  const scalar_t t_now = nodeHandle_->now().seconds();
  auto base_pose_m = pinocchioInterface_ptr_->getFramePose(base_name);

  // 初始化起始位置和姿态
  if (first_run) {
    first_run = false;
    pos_start = base_pose_m.translation();
    pos_start.z() = h_des;
    rpy_start = toEulerAngles(base_pose_m.rotation());
    rpy_start.head(2).setZero();
  } else {
    
    vector_t rpy_c = toEulerAngles(base_pose_m.rotation());
    if (computeEulerAngleErr(rpy_c, rpy_start).norm() < 0.3) {
      rpy_start += dt_ * vector3_t(0, 0, yawd_);
    }
    rpy_start.head(2).setZero();
    
    // 基于当前姿态和速度指令更新预测轨迹
    vector3_t vw = base_pose_m.rotation() * vel_cmd;  // 速度指令转换到全局坐标系
    /* if ((base_pose_m.translation() - pos_start).norm() < 0.2) {
      pos_start += dt_ * vw;
      pos_start.z() = h_des;
    } */
    pos_start.head(2) = base_pose_m.translation().head(2) + dt_ * vw.head(2);   // X-Y平面位置更新
    pos_start.z() = h_des;
  }

  // 生成时间序列和预测轨迹
  std::vector<scalar_t> time;
  std::vector<vector_t> rpy_t, pos_t;
  scalar_t horizon_time = referenceBuffer_->getModeSchedule()->duration();
  size_t N = horizon_time / 0.05;
  for (size_t k = 0; k < N; k++) {
    time.push_back(t_now + 0.05 * k);
    // 预测姿态（含偏航角速度）
    vector3_t rpy_k = rpy_start;
    rpy_k.z() += 0.05 * k * yawd_;
    rpy_t.emplace_back(rpy_k);
    // 预测位置（基于速度指令）
    vector3_t vel_des = toRotationMatrix(rpy_k) * vel_cmd;
    vector3_t pos_k = pos_start + 0.05 * k * vel_des;
    pos_k.z() = h_des;
    pos_t.emplace_back(pos_k);
  }

  // 三次样条插值生成基座位置和姿态轨迹
  auto cubicspline_pos = std::make_shared<CubicSplineTrajectory>(3);  // 位置轨迹
  cubicspline_pos->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      base_pose_m.rotation() * vel_cmd,
      CubicSplineInterpolation::BoundaryType::second_deriv, vector3_t::Zero());
  cubicspline_pos->fit(time, pos_t);
  referenceBuffer_->setIntegratedBasePosTraj(cubicspline_pos);

  auto cubicspline_rpy = std::make_shared<CubicSplineTrajectory>(3);
  cubicspline_rpy->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      vector3_t(0, 0, yawd_),
      CubicSplineInterpolation::BoundaryType::second_deriv, vector3_t::Zero());
  cubicspline_rpy->fit(time, rpy_t);
  referenceBuffer_->setIntegratedBaseRpyTraj(cubicspline_rpy);
}

// 动力学约束 （构建离散化状态空间方程）（dx/dt = Ax + Bu + b）
void ConvexMPC::getDynamics(scalar_t time_cur, size_t k,
                            const std::shared_ptr<ModeSchedule> mode_schedule) {
  const scalar_t time_k = time_cur + k * dt_; // 线速度积分项
  auto pos_traj = referenceBuffer_->getIntegratedBasePosTraj();
  auto rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();
  auto foot_traj = referenceBuffer_->getFootPosTraj();

  scalar_t phase = k * dt_ / mode_schedule->duration();
  auto contact_flag =
      biped::modeNumber2StanceLeg(mode_schedule->getModeFromPhase(phase));
  const size_t nf = foot_names.size();
  rcpputils::assert_true(nf == contact_flag.size());

  auto base_pose = pinocchioInterface_ptr_->getFramePose(base_name);
  vector3_t rpy_des = rpy_traj->evaluate(time_k);

  matrix_t Rt = toRotationMatrix(rpy_des);
  matrix_t Ig_t = Rt * Ig_ * Rt.transpose();

  // 状态转移矩阵 A 和输入矩阵 B。  状态方程 dx/dt = A x + B u
  /*
  A = [[I, dt*I,     0],
       [0, I, dt*J_rpy],
       [0, 0,        I]] (12x12)
  
  B = [[0        ],
       [dt*inv(M)],
       [0        ]]  (12x(3*nf))
  */
  ocp_[k].A.setIdentity(12, 12);  
  ocp_[k].A.block<3, 3>(0, 3).diagonal().fill(dt_);   // 位置积分
  ocp_[k].A.block<3, 3>(6, 9) = dt_ * getJacobiFromOmegaToRPY(rpy_des);   // 姿态积分
  ocp_[k].B.setZero(12, nf * 3);
  vector3_t xc = phase * pos_traj->evaluate(time_k) +
                 (1.0 - phase) * base_pose.translation();
  for (size_t i = 0; i < nf; i++) {
    const auto &foot_name = foot_names[i];
    if (contact_flag[i]) {
      vector3_t pf_i = foot_traj[foot_name]->evaluate(time_k);
      // 接触力对动力学的贡献
      ocp_[k].B.middleRows(3, 3).middleCols(3 * i, 3).diagonal().fill(
          dt_ / total_mass_);     // 力到加速度的映射
      ocp_[k].B.bottomRows(3).middleCols(3 * i, 3) =
          Ig_t.inverse() * skew(dt_ * (pf_i - xc));   // 重力补偿
    }
  }

  ocp_[k].b.setZero(12);
  ocp_[k].b(5) += -dt_ * grav_;
}


// 摩擦锥约束​​： 
void ConvexMPC::getInequalityConstraints(
    size_t k, size_t N, const std::shared_ptr<ModeSchedule> mode_schedule) {
  const size_t nf = foot_names.size();
  scalar_t phase = k * dt_ / mode_schedule->duration();
  auto contact_flag =
      biped::modeNumber2StanceLeg(mode_schedule->getModeFromPhase(phase));

  if (k < N) {
    scalar_t mu = 1 / mu_;  // 摩擦系数倒数
    matrix_t Ci(5, 3);  // 单足约束矩阵（摩擦锥模型）
    Ci << mu, 0, 1., -mu, 0, 1., 0, mu, 1., 0, -mu, 1., 0, 0, 1.;
    ocp_[k].C = matrix_t::Zero(5 * nf, 12);
    ocp_[k].D.setZero(5 * nf, 3 * nf);
    ocp_[k].lg.setZero(5 * nf);
    ocp_[k].ug.setZero(5 * nf);
    ocp_[k].lg_mask.setOnes(5 * nf);
    ocp_[k].ug_mask.setOnes(5 * nf);

    for (size_t i = 0; i < nf; i++) {
      ocp_[k].D.block<5, 3>(i * 5, i * 3) = Ci;
      ocp_[k].ug(5 * i + 4) = contact_flag[i] ? 400 : 0.0;
      ocp_[k].ug_mask.segment(5 * i, 4).setZero();
    }
  }
  /* std::cout << "\n############### " << k << " constraints ################\n"
           << cstr_k; */
}


void ConvexMPC::getCosts(scalar_t time_cur, size_t k, size_t N,
                         const std::shared_ptr<ModeSchedule> mode_schedule) {
  const size_t nf = foot_names.size();
  const scalar_t time_k = time_cur + (k + 1) * dt_;
  auto pos_traj = referenceBuffer_->getIntegratedBasePosTraj();
  auto rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();

  vector3_t rpy_des =
      rpy_traj->evaluate(time_k) - rpy_traj->evaluate(time_cur) + rpy_des_start;
  vector3_t omega_des =
      getJacobiFromRPYToOmega(rpy_des) * rpy_traj->derivative(time_k, 1);
  vector3_t v_des = pos_traj->derivative(time_k, 1);

  vector_t x_des(12);
  x_des << pos_traj->evaluate(time_k), v_des, rpy_des, omega_des;

  // std::cout << "xdes " << k << ": " << x_des.transpose() << "\n";


  // 二次型代价函数：x^T Q x + u^T R u + 2 x^T q + r^T u
  ocp_[k].Q = weight_;  // 状态误差权重
  // ocp_[k].S = matrix_t::Zero(3 * nf, 12);
  ocp_[k].q = -weight_ * x_des;   // 目标状态偏差惩罚
  ocp_[k].r.setZero(3 * nf);
  if (k < N) {
    ocp_[k].R = 1e-5 * matrix_t::Identity(3 * nf, 3 * nf);
    scalar_t phase = k * dt_ / mode_schedule->duration();
    auto contact_flag =
        biped::modeNumber2StanceLeg(mode_schedule->getModeFromPhase(phase));
    int nc = 0;
    for (bool flag : contact_flag) {
      if (flag) {
        nc++;
      }
    }
    nc = max(1, nc);
    vector3_t force_des_i =
        total_mass_ / nc *
        (vector3_t(0, 0, grav_) + pos_traj->derivative(time_k, 2));
    vector_t force_des = vector_t::Zero(3 * nf);
    for (size_t k = 0; k < nf; k++) {
      if (contact_flag[k]) {
        force_des.segment(3 * k, 3) = force_des_i;
      }
    }
    ocp_[k].r = -ocp_[k].R * force_des; // 控制输入对齐
  } else {
    // ocp_[k].Q = 1e2 * ocp_[k].Q;
    // ocp_[k].q = 1e2 * ocp_[k].q;
  }
}

void ConvexMPC::optimize() {
  const scalar_t time_cur = nodeHandle_->now().seconds();

  auto mode_schedule = referenceBuffer_->getModeSchedule();
  auto pos_traj = referenceBuffer_->getIntegratedBasePosTraj();
  auto rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();

  if (!mode_schedule || !pos_traj || !rpy_traj || referenceBuffer_->getFootPosTraj().empty()) {
    return;
  }

  const size_t N = mode_schedule->duration() / dt_;
  ocp_.resize(N + 1);
  solution_.resize(N + 1);

  matrix_t Ig_0 = pinocchioInterface_ptr_->getData().Ig.inertia().matrix();
  auto base_pose = pinocchioInterface_ptr_->getFramePose(base_name);
  auto base_twist = pinocchioInterface_ptr_->getFrame6dVel_localWorldAligned(base_name);
  Ig_ = base_pose.rotation().transpose() * Ig_0 * base_pose.rotation();

  auto rpy_m = toEulerAngles(base_pose.rotation());
  rpy_des_start = rpy_m - computeEulerAngleErr(rpy_m, rpy_traj->evaluate(time_cur));

  // x0 提前计算，供方案 C 的 δ₀ 使用
  vector_t x0(12);
  x0 << base_pose.translation(), base_twist.linear(), rpy_m, base_twist.angular();

  // MLP 残差补偿：两种模式
  // - Method C'（默认）：δ₀ = MLP(x₀, u_prev) 仅加到 b_0
  // - 全递归：每步 forwardSimulate 时调用 MLP(x_k, u_k)
  vector_t delta0 = vector_t::Zero(12);
  if (use_mlp_residual_ && !mlp_full_recursive_ && mlp_network_ && has_prev_data_) {
    // Method C'：单步扰动前馈
    vector_t mlp_input(x0.size() + prev_u_.size());
    mlp_input << x0, prev_u_;
    delta0 = mlp_network_->forward(mlp_input);
    delta0 = delta0.cwiseMax(-mlp_residual_clamp_).cwiseMin(mlp_residual_clamp_);
  }

  for (size_t k = 0; k <= N; ++k) {
    if (k < N) {
      getDynamics(time_cur, k, mode_schedule);
      if (k == 0 && !mlp_full_recursive_) {
        ocp_[0].b += delta0;  // Method C'：只修正第一步
      }
    }
    getInequalityConstraints(k, N, mode_schedule);
    getCosts(time_cur, k, N, mode_schedule);
  }

  const size_t nx = 12;
  const size_t nu = foot_names.size() * 3;

  PMPOptimizer optimizer(ocp_, nx, nu);
  optimizer.setInitialState(x0);
  if (!solution_.empty()) {
    optimizer.setWarmStart(solution_);
  }

  // 全递归 MLP：在 forwardSimulate 每步调用 MLP 修正动力学
  if (use_mlp_residual_ && mlp_full_recursive_ && mlp_network_) {
    double clamp = mlp_residual_clamp_;
    int max_steps = mlp_recursive_steps_;  // -1 = 全部 N 步
    auto* mlp_ptr = mlp_network_.get();
    optimizer.setResidualCallback(
      [mlp_ptr, clamp, max_steps](size_t k, const vector_t& x, const vector_t& u) -> vector_t {
        if (max_steps >= 0 && static_cast<int>(k) >= max_steps) {
          return vector_t::Zero(x.size());  // 超过指定步数，不修正
        }
        vector_t mlp_input(x.size() + u.size());
        mlp_input << x, u;
        vector_t delta = mlp_ptr->forward(mlp_input);
        return delta.cwiseMax(-clamp).cwiseMin(clamp);
      });
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // 数据采集：利用当前 x0 作为上一步的 x_next_true
  if (collect_training_data_) {
    collectTrainingData(x0);
  }

  if (optimizer.solve(solution_)) {
    auto end_time = std::chrono::high_resolution_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // 每步都更新 prev_u_ / has_prev_data_，方案 C 需要上一步控制量
    // 注意：此更新必须独立于 collect_training_data_，否则 use_mlp_residual_ 时
    //       has_prev_data_ 永远为 false，delta0 始终为零，方案 C 不生效。
    prev_u_ = solution_[0].u;
    has_prev_data_ = true;

    // 训练数据采集（额外记录 x_k, x_next_est, delta_mlp）
    if (collect_training_data_ && !training_data_written_) {
      prev_x_ = x0;
      // 记录名义预测（去除 delta0，保证训练数据格式与无 MLP 时一致）
      prev_x_next_est_ = ocp_[0].A * x0 + ocp_[0].B * solution_[0].u + (ocp_[0].b - delta0);
      // 影子 MLP：计算 delta_mlp(x0, u0) 但不施加到控制
      if (mlp_network_) {
        vector_t mlp_input(x0.size() + solution_[0].u.size());
        mlp_input << x0, solution_[0].u;
        prev_delta_mlp_ = mlp_network_->forward(mlp_input);
        prev_delta_mlp_ = prev_delta_mlp_.cwiseMax(-mlp_residual_clamp_)
                                         .cwiseMin( mlp_residual_clamp_);
      } else {
        prev_delta_mlp_.setZero(12);
      }
    }

    fitTraj(time_cur, N);
    evaluateAndExportMetrics(time_cur, dt_, solve_ms);
  }
}

void ConvexMPC::fitTraj(scalar_t time_cur, size_t N) {
  std::vector<scalar_t> time_array;
  std::vector<vector_t> base_pos_array;
  std::vector<vector_t> base_vel_array;
  std::vector<vector_t> base_rpy_array;
  std::vector<vector_t> base_omega_array;
  std::vector<vector_t> force_array;

  matrix_t P(6, 12);
  P.setZero();
  P.topRows(3).middleCols(3, 3).setIdentity();
  P.bottomRows(3).middleCols(9, 3).setIdentity();

  matrix_t A = 1.0 / dt_ * (ocp_[0].A - matrix_t::Identity(12, 12));
  matrix_t B = 1.0 / dt_ * ocp_[0].B;
  vector_t drift = 1.0 / dt_ * ocp_[0].b;
  vector_t acc_des = P * (A * solution_[0].x + B * solution_[0].u + drift);

  for (size_t k = 0; k < N; k++) {
    time_array.emplace_back(time_cur + k * dt_);
    base_pos_array.emplace_back(solution_[k].x.head(3));
    base_vel_array.emplace_back(solution_[k].x.segment(3, 3));
    base_rpy_array.emplace_back(solution_[k].x.segment(6, 3));
    base_omega_array.emplace_back(solution_[k].x.tail(3));
    force_array.emplace_back(solution_[k].u);
  }

  auto base_pos_traj_ptr_ = std::make_shared<CubicSplineTrajectory>(
      3, CubicSplineInterpolation::SplineType::cspline);
  base_pos_traj_ptr_->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      solution_[1].x.segment(3, 3),
      CubicSplineInterpolation::BoundaryType::first_deriv,
      solution_.back().x.segment(3, 3));
  base_pos_traj_ptr_->fit(time_array, base_pos_array);
  referenceBuffer_->setOptimizedBasePosTraj(base_pos_traj_ptr_);

  auto base_vel_traj_ptr_ = std::make_shared<CubicSplineTrajectory>(
      3, CubicSplineInterpolation::SplineType::cspline);
  base_vel_traj_ptr_->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ * (solution_[1].x.segment(3, 3) - solution_[0].x.segment(3, 3)),
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ * (solution_[N - 1].x.segment(3, 3) - solution_[N - 2].x.segment(3, 3)));
  base_vel_traj_ptr_->fit(time_array, base_vel_array);
  referenceBuffer_->setOptimizedBaseVelTraj(base_vel_traj_ptr_);

  auto base_rpy_traj_ptr_ = std::make_shared<CubicSplineTrajectory>(
      3, CubicSplineInterpolation::SplineType::cspline_hermite);
  base_rpy_traj_ptr_->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ * (solution_[1].x.segment(6, 3) - solution_[0].x.segment(6, 3)),
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ *
          (solution_[N - 1].x.segment(6, 3) -
           solution_[N - 2].x.segment(6, 3)));
  base_rpy_traj_ptr_->fit(time_array, base_rpy_array);
  referenceBuffer_->setOptimizedBaseRpyTraj(base_rpy_traj_ptr_);

  auto base_omega_traj_ptr_ = std::make_shared<CubicSplineTrajectory>(
      3, CubicSplineInterpolation::SplineType::cspline);
  base_omega_traj_ptr_->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ * (solution_[1].x.tail(3) - solution_[0].x.tail(3)),
      CubicSplineInterpolation::BoundaryType::first_deriv,
      1.0 / dt_ * (solution_[N - 1].x.tail(3) - solution_[N - 2].x.tail(3)));
  base_omega_traj_ptr_->fit(time_array, base_omega_array);
  referenceBuffer_->setOptimizedBaseOmegaTraj(base_omega_traj_ptr_);

  auto force_traj_ptr_ = std::make_shared<CubicSplineTrajectory>(
      foot_names.size() * 3, CubicSplineInterpolation::SplineType::cspline);
  force_traj_ptr_->set_boundary(
      CubicSplineInterpolation::BoundaryType::first_deriv,
      vector_t::Zero(foot_names.size() * 3),
      CubicSplineInterpolation::BoundaryType::first_deriv,
      vector_t::Zero(foot_names.size() * 3));
  force_traj_ptr_->fit(time_array, force_array);
  referenceBuffer_->setOptimizedForceTraj(force_traj_ptr_);
}

void ConvexMPC::evaluateAndExportMetrics(double t0, double dt, double solve_time_ms) {
  double t_now = nodeHandle_->now().seconds();
  static std::vector<std::string> metrics_cache_;
  static bool has_written_csv_ = false;

  if (!has_written_csv_ && t_now - time_prog_begin > 20.0) {
    std::ofstream file("pmp_ml_eval_result.csv");
    file << "timestamp,max_delta_u,avg_delta_u,max_tracking_error,avg_tracking_error,max_constraint_violation,solve_time_ms\n";
    for (const auto& line : metrics_cache_) {
      file << line << "\n";
    }
    file.close();
    has_written_csv_ = true;
    RCLCPP_INFO(nodeHandle_->get_logger(), "[Eval] CSV written: pmp_ml_eval_result.csv (%zu rows)", metrics_cache_.size());
    return;
  }
  if (has_written_csv_) return;

  const size_t N = solution_.size() - 1;
  if (N == 0) return;

  auto pos_traj = referenceBuffer_->getIntegratedBasePosTraj();
  auto rpy_traj = referenceBuffer_->getIntegratedBaseRpyTraj();
  if (!pos_traj || !rpy_traj) return;

  // Compute max/avg delta_u
  scalar_t max_delta_u = 0.0;
  scalar_t sum_delta_u = 0.0;
  for (size_t k = 1; k < N; ++k) {
    scalar_t du = (solution_[k].u - solution_[k-1].u).lpNorm<Eigen::Infinity>();
    max_delta_u = std::max(max_delta_u, du);
    sum_delta_u += du;
  }
  scalar_t avg_delta_u = (N > 1) ? sum_delta_u / (N - 1) : 0.0;

  // Compute tracking error (12D)
  scalar_t max_tracking_error = 0.0;
  scalar_t sum_tracking_error = 0.0;
  for (size_t k = 0; k <= N; ++k) {
    const scalar_t time_k = t0 + (k + 1) * dt;
    vector3_t rpy_des = rpy_traj->evaluate(time_k) - rpy_traj->evaluate(t0) + rpy_des_start;
    vector3_t omega_des = getJacobiFromRPYToOmega(rpy_des) * rpy_traj->derivative(time_k, 1);
    vector3_t v_des = pos_traj->derivative(time_k, 1);
    vector_t x_des(12);
    x_des << pos_traj->evaluate(time_k), v_des, rpy_des, omega_des;

    scalar_t err = (solution_[k].x - x_des).lpNorm<Eigen::Infinity>();
    max_tracking_error = std::max(max_tracking_error, err);
    sum_tracking_error += err;
  }
  scalar_t avg_tracking_error = sum_tracking_error / (N + 1);

  // Compute constraint violation
  scalar_t max_constraint_violation = 0.0;
  for (size_t k = 0; k < N; ++k) {
    if (ocp_[k].C.size() == 0 || ocp_[k].D.size() == 0) continue;
    if (ocp_[k].ug.size() == 0) continue;
    vector_t gk = ocp_[k].C * solution_[k].x + ocp_[k].D * solution_[k].u;
    vector_t delta_ug = (gk - ocp_[k].ug).cwiseMax(0.0);
    scalar_t violation = delta_ug.maxCoeff();
    max_constraint_violation = std::max(max_constraint_violation, violation);
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6)
      << t_now << ","
      << max_delta_u << "," << avg_delta_u << ","
      << max_tracking_error << "," << avg_tracking_error << ","
      << max_constraint_violation << ","
      << solve_time_ms;
  metrics_cache_.push_back(oss.str());
}


// ─── 训练数据采集 ──────────────────────────────────────────────
// 调用时机：每步 optimize() 开头，x_now 即上一步的 x_next_true
void ConvexMPC::collectTrainingData(const vector_t& x_now) {
  if (training_data_written_ || !has_prev_data_) return;
  if ((int)training_rows_.size() >= max_training_samples_) {
    writeTrainingData();
    return;
  }

  // 组装一行: timestamp, x_k(12), u_k(6), x_next_est(12), x_next_true(12), delta_mlp(12)
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << nodeHandle_->now().seconds();
  for (int i = 0; i < 12; ++i) oss << "," << prev_x_(i);
  for (int i = 0; i < (int)prev_u_.size(); ++i) oss << "," << prev_u_(i);
  for (int i = 0; i < 12; ++i) oss << "," << prev_x_next_est_(i);
  for (int i = 0; i < 12; ++i) oss << "," << x_now(i);
  // 影子 MLP 预测（不施加，仅记录）：delta_true = x_now - x_next_est 可事后计算
  if (prev_delta_mlp_.size() == 0) prev_delta_mlp_.setZero(12);
  for (int i = 0; i < 12; ++i) oss << "," << prev_delta_mlp_(i);
  training_rows_.push_back(oss.str());

  if ((int)training_rows_.size() % 1000 == 0) {
    RCLCPP_INFO(nodeHandle_->get_logger(),
                "[TrainData] Collected %zu / %d samples",
                training_rows_.size(), max_training_samples_);
  }
}

void ConvexMPC::writeTrainingData() {
  if (training_data_written_ || training_rows_.empty()) return;

  // 生成带时间戳的文件名
  auto t = std::chrono::system_clock::now().time_since_epoch().count();
  std::string path = "/home/markwcl/mpc_data/mlp_training_data_" +
                     std::to_string(t / 1000000000LL) + ".csv";

  std::ofstream f(path);
  // 表头
  f << "timestamp";
  for (int i = 0; i < 12; ++i) f << ",x_n_" << i;
  for (int i = 0; i < (int)prev_u_.size(); ++i) f << ",u_n_" << i;
  for (int i = 0; i < 12; ++i) f << ",x_next_est_" << i;
  for (int i = 0; i < 12; ++i) f << ",x_next_true_" << i;
  for (int i = 0; i < 12; ++i) f << ",delta_mlp_" << i;
  f << "\n";
  for (auto& row : training_rows_) f << row << "\n";
  f.close();

  training_data_written_ = true;
  RCLCPP_INFO(nodeHandle_->get_logger(),
              "[TrainData] Saved %zu samples -> %s",
              training_rows_.size(), path.c_str());
}

} // namespace clear
