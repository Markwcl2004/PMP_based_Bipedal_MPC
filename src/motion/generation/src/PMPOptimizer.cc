// PMPOptimizer.cc - Pontryagin-based optimizer (FBDE + superlinear iteration)

#include "generation/PMPOptimizer.h"
#include <iostream>

namespace clear {

PMPOptimizer::PMPOptimizer(const std::vector<OcpStage>& ocp, size_t nx, size_t nu)
    : ocp_(ocp), nx_(nx), nu_(nu), N_(ocp.size() - 1) {}

void PMPOptimizer::setInitialState(const vector_t& x0) {
  x0_ = x0;
}

void PMPOptimizer::setResidualCallback(ResidualFn fn) {
  residual_fn_ = std::move(fn);
}

void PMPOptimizer::setWarmStart(const std::vector<OcpSolution>& prev_solution) {
  size_t prev_N = prev_solution.size() - 1;
  if (prev_N == 0 || prev_N != N_) {
    has_warm_start_ = false;
    return;
  }
  if (prev_solution[0].u.size() != static_cast<int>(nu_)) {
    has_warm_start_ = false;
    return;
  }
  warm_u_.resize(N_);
  for (size_t k = 0; k < N_ - 1; ++k) {
    warm_u_[k] = prev_solution[k + 1].u;
  }
  warm_u_[N_ - 1] = prev_solution[N_ - 1].u;
  has_warm_start_ = true;
}

/*
  Attention!!!
  "stages_[k].lambda" means lambda_{k+1} in the essay (k = 0, ..., N)
*/

void PMPOptimizer::initializeTrajectory() {
  stages_.clear();
  stages_.resize(N_ + 1);
  for (size_t k = 0; k <= N_; ++k) {
    stages_[k].x = (k == 0 ? x0_ : vector_t::Zero(nx_));
    if (has_warm_start_ && k < N_) {
      stages_[k].u = warm_u_[k];
    } else {
      stages_[k].u = vector_t::Zero(nu_);
    }
    stages_[k].lambda = vector_t::Zero(nx_);
  }
}

void PMPOptimizer::forwardSimulate() {
  for (size_t k = 0; k < N_; ++k) {
    const auto& A = ocp_[k].A;
    const auto& B = ocp_[k].B;
    const auto& b = ocp_[k].b;
    stages_[k + 1].x = A * stages_[k].x + B * stages_[k].u + b;
    // RTI: MLP 残差补偿 (Level 1)
    if (residual_fn_) {
      stages_[k + 1].x += residual_fn_(stages_[k].x, stages_[k].u);
    }
  }
}

void PMPOptimizer::backwardCostate() {
  stages_[N_].lambda.setZero();  // λ_{N+1} = 0

  // 终端协态: λ_N = Q_N * x_N + q_N  (匹配 HPIPM 的 (1/2)x^T Q x + q^T x 约定)
  // 按约定 stages_[k].lambda = λ_{k+1}，所以 λ_N 存在 stages_[N_-1].lambda
  const auto& Q_N = ocp_[N_].Q;
  const auto& q_N = ocp_[N_].q;
  stages_[N_ - 1].lambda = Q_N * stages_[N_].x + q_N;

  for (int k = N_ - 1; k >= 1; --k) {
    const auto& A = ocp_[k].A;
    const auto& Q = ocp_[k].Q;
    const auto& q = ocp_[k].q;

    // λ_k = Q_k * x_k + q_k + A_k^T * λ_{k+1}
    stages_[k - 1].lambda = Q * stages_[k].x
                            + q + A.transpose() * stages_[k].lambda;
  }
}

vector_t PMPOptimizer::computeGradient() {
    vector_t grad = vector_t::Zero(N_ * nu_);

    for (size_t k = 0; k < N_; ++k) {
        const auto& R = ocp_[k].R;
        const auto& r = ocp_[k].r;
        const auto& B = ocp_[k].B;

        // PMP 一阶最优性条件: ∂H/∂u_k = R_k u_k + r_k + B_k^T λ_{k+1}  (匹配 HPIPM 的 (1/2) 约定)
        grad.segment(k * nu_, nu_) = R * stages_[k].u
                                      + r + B.transpose() * stages_[k].lambda;
    }

    return grad;
}

// ============================================================
// 高效 Hessian 计算：使用反向代价矩阵 S (block 级别)
// 匹配 HPIPM 的 (1/2) 约定: cost = (1/2)x^T Q x + q^T x + (1/2)u^T R u + r^T u
//
//   S[k] = Σ_{m=k}^{N} Φ(m,k)^T Q_m Φ(m,k)
//
// 对角块:   H_{ii} = R_i + B_i^T S[i+1] B_i
// 非对角块: H_{ij} = B_i^T Φ(j+1,i+1)^T S[j+1] B_j  (i < j)
// ============================================================

matrix_t PMPOptimizer::computeHessian() {
    const size_t N = N_;
    const size_t nx = nx_;
    const size_t nu = nu_;

    matrix_t H = matrix_t::Zero(N * nu, N * nu);

    // 反向递推 S[k]
    std::vector<matrix_t> S(N + 1, matrix_t::Zero(nx, nx));
    S[N] = ocp_[N].Q;
    for (int k = static_cast<int>(N) - 1; k >= 0; --k) {
        S[k] = ocp_[k].Q + ocp_[k].A.transpose() * S[k + 1] * ocp_[k].A;
    }

    // 对角块: H_{ii} = R_i + B_i^T S[i+1] B_i
    for (size_t i = 0; i < N; ++i) {
        H.block(i * nu, i * nu, nu, nu) =
            ocp_[i].R + ocp_[i].B.transpose() * S[i + 1] * ocp_[i].B;
    }

    // 非对角块 (i < j):
    // Φ(j+1, i+1) 通过逐步左乘 A_j 累积
    for (size_t i = 0; i < N; ++i) {
        matrix_t Phi = matrix_t::Identity(nx, nx);  // Φ(i+1, i+1) = I
        for (size_t j = i + 1; j < N; ++j) {
            Phi = ocp_[j].A * Phi;  // Phi = Φ(j+1, i+1)
            matrix_t block = ocp_[i].B.transpose() * Phi.transpose() * S[j + 1] * ocp_[j].B;
            H.block(i * nu, j * nu, nu, nu) = block;
            H.block(j * nu, i * nu, nu, nu) = block.transpose();
        }
    }

    return H;
}

vector_t PMPOptimizer::superlinearUpdate(const vector_t& grad, const matrix_t& hess) {
  const size_t N = N_;
  const size_t nu = nu_;
  const size_t dim = N * nu;
  const int max_iters = 30;
  const double epsilon = 1e-8;

  // block 对角 R_big = diag(R_0, ..., R_{N-1})
  matrix_t R_big = matrix_t::Zero(dim, dim);
  for (size_t k = 0; k < N; ++k) {
    R_big.block(k * nu, k * nu, nu, nu) = ocp_[k].R;
  }

  // H_aug = R + Hess + εI
  matrix_t H_aug = R_big + hess + epsilon * matrix_t::Identity(dim, dim);
  H_aug = 0.5 * (H_aug + H_aug.transpose());

  Eigen::LLT<matrix_t> llt(H_aug);
  if (llt.info() != Eigen::Success) {
    std::cerr << "[PMPOptimizer] Cholesky decomposition failed!\n";
    return vector_t::Zero(dim);
  }

  // 超线性迭代: 收敛到 g* = (H + εI)^{-1} grad
  vector_t g = llt.solve(-grad);
  for (int j = 0; j < max_iters; ++j) {
    g = llt.solve(grad + R_big * g);
  }

  return g;
}

void PMPOptimizer::updateControl(const vector_t& g) {
  for (size_t k = 0; k < N_; ++k) {
    stages_[k].u -= g.segment(k * nu_, nu_);
  }
}

// ============================================================
// PMP 求解流程 (匹配 HPIPM 的 (1/2) 代价约定):
//   ① 初始化轨迹 (含热启动)
//   ② 前向传播状态 x
//   ③ 反向计算协态 λ
//   ④ 计算梯度 ∂H/∂u
//   ⑤ 计算 Hessian ∂²J/∂u² (反向代价矩阵法)
//   ⑥ 超线性迭代求牛顿方向
//   ⑦ 更新控制量
//   ⑧ 重新前向传播 (保证 x 与 u 一致)
// ============================================================

bool PMPOptimizer::solve(std::vector<OcpSolution>& solution_out) {
  // ① 初始化
  initializeTrajectory();

  // ② 前向传播
  forwardSimulate();

  // ③ 反向协态
  backwardCostate();

  // ④ 计算梯度
  vector_t grad = computeGradient();
  if (!grad.allFinite()) {
    std::cerr << "[PMPOptimizer] gradient contains NaN/Inf, aborting.\n";
    return false;
  }

  // ⑤ 计算 Hessian
  matrix_t hess = computeHessian();

  // ⑥ 超线性更新
  vector_t update = superlinearUpdate(grad, hess);

  // ⑦ 更新控制量
  updateControl(update);

  // ⑧ 用更新后的 u 重新前向传播，保证 x 与 u 一致
  forwardSimulate();

  // 输出
  solution_out.resize(N_ + 1);
  for (size_t i = 0; i <= N_; ++i) {
    solution_out[i].x = stages_[i].x;
    solution_out[i].u = stages_[i].u;
    solution_out[i].pi = stages_[i].lambda;
    solution_out[i].P = Eigen::MatrixXd::Identity(nx_, nx_);
    solution_out[i].p = Eigen::VectorXd::Zero(nx_);
    solution_out[i].K = Eigen::MatrixXd::Zero(nu_, nx_);
    solution_out[i].k = Eigen::VectorXd::Zero(nu_);
  }

  return true;
}

// 以下为原始逐标量 beta/alpha 方法 (已被 block 级别 Hessian 替代，保留备参考)

vector_t PMPOptimizer::computeBetaTerm(size_t k, size_t i) const {
    size_t nx = nx_;
    size_t N = N_;
    std::vector<vector_t> beta(N + 1, vector_t::Zero(nx));
    for (int j = static_cast<int>(N) - 1; j >= static_cast<int>(k); --j) {
        const auto& A = ocp_[j].A;
        const auto& B = ocp_[j].B;
        size_t control_stage = i / nu_;
        size_t control_idx = i % nu_;
        if (j == static_cast<int>(control_stage)) {
            beta[j] = B.col(control_idx) + A.transpose() * beta[j + 1];
        } else {
            beta[j] = A.transpose() * beta[j + 1];
        }
    }
    return beta[k];
}

vector_t PMPOptimizer::computeAlphaTerm(size_t k, size_t i) const {
    size_t nx = nx_;
    size_t N = N_;
    std::vector<vector_t> alpha(N + 1, vector_t::Zero(nx));
    for (int j = static_cast<int>(N) - 1; j >= static_cast<int>(k); --j) {
        const auto& A = ocp_[j].A;
        const auto& Q = ocp_[j].Q;
        const matrix_t& Qt = Q;  // 匹配 HPIPM 的 (1/2) 约定
        const vector_t beta_j = computeBetaTerm(j, i);
        alpha[j] = A.transpose() * alpha[j + 1] + Qt * beta_j;
    }
    return alpha[k];
}

} // namespace clear
