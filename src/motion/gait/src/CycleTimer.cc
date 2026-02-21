#include "gait/CycleTimer.h"
#include <core/misc/NumericTraits.h>

/*
步态周期计时器，用于跟踪当前步态周期内的时间进度。

innerLoop: 内部循环，更新当前周期时间。
    控制循环的速率，使其保持在每秒 10000 次。
    调整周期起始点，确保其始终处于当前时间之前，并且与周期时间对齐。
    计算当前周期的时间，并将其存储在 current_cycle_time_ 中。
    检查 ROS 节点是否正在运行，并根据 run_ 的值决定是否继续循环。

getCycleTime: 获取当前周期内的绝对时间。
timerReset: 重置计时器。
*/

namespace clear {
CycleTimer::CycleTimer(Node::SharedPtr nodeHandle, scalar_t cycle_duration)
    : nodeHandle_(nodeHandle), cycle_duration_(cycle_duration) {
  if (cycle_duration_ < numeric_traits::limitEpsilon<scalar_t>()) {
    throw std::out_of_range("cycle_duration less than 0");
  }
  cycle_start_point_.push(nodeHandle_->now().seconds());
  current_cycle_time_.push(0.0);
  inner_loop_thread_ = std::thread(&CycleTimer::innerLoop, this);
  run_.push(true);
}

CycleTimer::~CycleTimer() {
  run_.push(false);
  inner_loop_thread_.join();
}

void CycleTimer::innerLoop() {
  rclcpp::Rate loop_rate(10000.0);
  while (rclcpp::ok() && run_.get()) {
    while (cycle_start_point_.get() <
           nodeHandle_->now().seconds() - cycle_duration_) {
      cycle_start_point_.push(cycle_start_point_.get() + cycle_duration_);
    }
    if (cycle_start_point_.get() > nodeHandle_->now().seconds()) {
      cycle_start_point_.push(nodeHandle_->now().seconds());
    }
    current_cycle_time_.push(nodeHandle_->now().seconds() -
                             cycle_start_point_.get());
    loop_rate.sleep();
  }
}

void CycleTimer::timerReset() {
  cycle_start_point_.push(nodeHandle_->now().seconds());
}

scalar_t CycleTimer::getCycleTime() { return current_cycle_time_.get(); }

} // namespace clear
