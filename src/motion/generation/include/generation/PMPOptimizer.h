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
  // B3 AL-full (codex collapse-review verdict V2): augmented-Lagrangian
  // gradient/Hessian penalty for the friction inequalities. Constraints
  // enter the PMP Newton system through first- and second-order penalty
  // terms while the post-solve projection is retained as hard-feasibility
  // safety. Default flag off → legacy projection-only path unchanged.
  void setUseAugmentedLagrangian(bool enable) { use_augmented_lagrangian_ = enable; }
  void setALOptions(double rho,
                    int outer_iters,
                    bool final_projection_safety,
                    double active_threshold);
  // B3 Fix A (codex full-review V2): true AL warm-start. ConvexMPC owns the
  // persistent ν containers and hands them in before each solve(); after
  // solve() the caller fetches the updated ν back via getALMultipliers().
  // Without this, every tick rebuilds optimizer and ν is reset to 0 → only
  // a one-shot penalty, not a real Augmented-Lagrangian outer iteration.
  void setALMultipliers(const std::vector<vector_t>& nu_upper,
                        const std::vector<vector_t>& nu_lower);
  void getALMultipliers(std::vector<vector_t>& nu_upper,
                        std::vector<vector_t>& nu_lower) const;
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
  void updateControl(const vector_t& g, bool project_after_update = true);
  void projectConstraints();

  // B3 AL-full helpers.
  //   addAugmentedLagrangianPenalty: adds D^T η to grad and ρ D_A^T D_A to
  //   the block-diagonal of hess for the current multipliers + active rows.
  //   updateMultipliers: ν_k ← [ν_k + ρ c_k(u_k)]_+ after each PMP solve.
  //   stageRowResidual: returns c_{k,j}(u_k) = D_{k,j} u_k - ug_{k,j} (or
  //   lg-D, depending on row sign).
  void addAugmentedLagrangianPenalty(vector_t& grad, matrix_t& hess);
  void updateALMultipliers();

  std::vector<OcpStage> ocp_;
  std::vector<PMPStageData> stages_;
  vector_t x0_;
  size_t nx_, nu_, N_;

  bool has_warm_start_ = false;
  std::vector<vector_t> warm_u_;

  ResidualFn residual_fn_;

  // B3 AL-full state. All zero/empty by default → legacy projection-only path
  // unchanged (regression-safe).
  bool use_augmented_lagrangian_ = false;
  double al_rho_ = 1.0e-2;             // penalty strength
  int al_outer_iters_ = 1;             // outer multiplier updates per solve()
  bool al_final_projection_safety_ = true;
  double al_active_threshold_ = 0.0;   // active if (c + ν/ρ) > threshold
  // Lagrange multipliers per stage per constraint row. Persisted across
  // ticks so warm-starting works (ν is a slow variable, not reset each tick).
  std::vector<vector_t> al_nu_upper_;  // size N, each row count = D.rows()
  std::vector<vector_t> al_nu_lower_;
};

} // namespace clear
