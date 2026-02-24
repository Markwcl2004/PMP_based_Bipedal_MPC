#pragma once

#include <vector>
#include <functional>
#include <Eigen/Dense>

namespace clear {

// === 类型定义 ===
using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;


// === 单步优化结构 ===
struct OcpStage {
  matrix_t A, B, Q, R, C, D;
  vector_t b, q, r, lg, ug;
  Eigen::VectorXi lg_mask, ug_mask;
};

// === 优化结果结构 ===
struct OcpSolution {
  vector_t x;      // 状态
  vector_t u;      // 控制输入
  vector_t pi;     // 协态（costate）
  matrix_t P;      // Riccati矩阵（可选，用于策略反馈）
  vector_t p;      // Riccati向量
  matrix_t K;      // 反馈增益
  vector_t k;      // 前馈增益
};

//wcl
// std::vector<OcpStage> pmp_ocp_;
// std::vector<OcpSolution> pmp_solution_;

// === FBDE阶段结构 ===
struct PMPStageData {
  vector_t x, u, lambda;

  matrix_t P;  // nx × nx
  vector_t p;  // nx

  matrix_t K;  // nu × nx
  vector_t k;  // nu

  
};

// === PMP优化器主类 ===
class PMPOptimizer {
 public:
  using ResidualFn = std::function<vector_t(size_t k, const vector_t& x, const vector_t& u)>;

  PMPOptimizer(const std::vector<OcpStage>& ocp, size_t nx, size_t nu);
  void setInitialState(const vector_t& x0);
  void setWarmStart(const std::vector<OcpSolution>& prev_solution);
  void setResidualCallback(ResidualFn fn);
  bool solve(std::vector<OcpSolution>& solution_out);

 private:
  double rho_ = 780.0;
  void initializeTrajectory();
  void forwardSimulate();
  void backwardCostate();
  vector_t computeGradient();

  vector_t computeBetaTerm(size_t k, size_t i) const;
  vector_t computeAlphaTerm(size_t k, size_t i) const;
  matrix_t computeHessian();
  // vector_t computeAlphaTerm(size_t k);
  // vector_t computeBetaTerm(size_t k, const vector_t& beta_k);
  // matrix_t computeHessian();

  vector_t superlinearUpdate(const vector_t& grad, const matrix_t& hess);
  void updateControl(const vector_t& g);

  std::vector<OcpStage> ocp_;
  std::vector<PMPStageData> stages_;
  vector_t x0_;
  size_t nx_, nu_, N_;

  bool has_warm_start_ = false;
  std::vector<vector_t> warm_u_;

  ResidualFn residual_fn_;
};

} // namespace clear