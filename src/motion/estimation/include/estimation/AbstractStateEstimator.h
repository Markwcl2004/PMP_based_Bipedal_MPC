#pragma once

#include <memory>
#include <vector>
#include <core/types.h>
#include <core/gait/ModeSchedule.h>

namespace clear {

/**
 * @brief 状态估计器类型枚举
 */
enum class EstimatorType {
  LKF,      ///< 线性卡尔曼滤波器
  MLP,      ///< 多层感知机
  RNN,      ///< 循环神经网络
  LSTM,     ///< 长短期记忆网络 (预留)
  GRU       ///< 门控循环单元 (预留)
};

/**
 * @brief 抽象状态估计器基类
 *
 * 定义了所有状态估计器的通用接口，包括：
 * - 状态查询 (getQpos, getQvel)
 * - 参数设置 (setContactFlag, updateModeSchedule)
 * - 初始化 (setup)
 *
 * 具体实现类需要继承此基类并实现所有纯虚函数。
 */
class AbstractStateEstimator {
public:
  virtual ~AbstractStateEstimator() = default;

  /**
   * @brief 初始化估计器
   *
   * 设置初始状态、协方差矩阵、系统参数等
   */
  virtual void setup() = 0;

  /**
   * @brief 获取当前位姿估计
   * @return 位姿向量 (位置 + 姿态四元数 + 关节位置)
   */
  virtual std::shared_ptr<vector_t> getQpos() = 0;

  /**
   * @brief 获取当前速度估计
   * @return 速度向量 (线速度 + 角速度 + 关节速度)
   */
  virtual std::shared_ptr<vector_t> getQvel() = 0;

  /**
   * @brief 更新步态模式调度
   * @param mode_schedule 步态模式调度信息
   */
  virtual void updateModeSchedule(std::shared_ptr<ModeSchedule> mode_schedule) = 0;

  /**
   * @brief 设置足部接触标志
   * @param flag 接触标志向量 (true表示接触地面)
   */
  virtual void setContactFlag(std::vector<bool> flag) = 0;

  /**
   * @brief 获取估计器类型
   * @return 估计器类型枚举
   */
  virtual EstimatorType getType() const = 0;

  /**
   * @brief 获取估计器类型的字符串表示
   * @return 类型名称
   */
  virtual std::string getTypeName() const {
    switch (getType()) {
      case EstimatorType::LKF:  return "LKF";
      case EstimatorType::MLP:  return "MLP";
      case EstimatorType::RNN:  return "RNN";
      case EstimatorType::LSTM: return "LSTM";
      case EstimatorType::GRU:  return "GRU";
      default: return "Unknown";
    }
  }
};

} // namespace clear
