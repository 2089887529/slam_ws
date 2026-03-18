#!/usr/bin/env python3
"""
Bag 质量检测脚本
用法：
  终端1: ros2 bag play your_bag/
  终端2: ros2 run bag_quality_checker bag_quality_check
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
import sensor_msgs_py.point_cloud2 as pc2
import math
import sys
import signal
import time
import collections

# ==================== 配置 ====================
LIDAR_TOPIC = "/rslidar_points"
IMU_TOPIC   = "/rslidar_imu_data"

# 期望频率
LIDAR_HZ_EXPECTED = 10.0
IMU_HZ_EXPECTED   = 100.0
HZ_WARN_RATIO     = 0.8     # 低于期望 80% 告警

# 判断阈值
NAN_RATIO_WARN      = 0.3
MIN_POINTS_WARN     = 5000
STAMP_GAP_WARN      = 0.3
IMU_ACC_WARN        = 50.0
IMU_GYR_WARN        = 10.0
IMU_STAMP_GAP_WARN  = 0.02   # 20ms

# bag 播放结束检测
TIMEOUT_SEC = 3.0
# =============================================

RESET  = "\033[0m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


class BagQualityChecker(Node):

    def __init__(self):
        super().__init__('bag_quality_checker')

        # 点云统计
        self.lidar_count       = 0
        self.lidar_bad_nan     = 0
        self.lidar_bad_points  = 0
        self.lidar_bad_stamp   = 0
        self.last_lidar_stamp  = None
        self.lidar_nan_max     = 0.0
        self.lidar_hz_warn     = 0
        self.lidar_hz_min      = float('inf')
        self.lidar_hz_max      = 0.0
        self.lidar_ts_window   = collections.deque(maxlen=50)

        # IMU 统计
        self.imu_count         = 0
        self.imu_bad_acc       = 0
        self.imu_bad_gyr       = 0
        self.imu_bad_stamp     = 0
        self.last_imu_stamp    = None
        self.imu_hz_warn       = 0
        self.imu_hz_min        = float('inf')
        self.imu_hz_max        = 0.0
        self.imu_ts_window     = collections.deque(maxlen=200)

        # bag 时间范围
        self.bag_start_stamp   = None
        self.bag_end_stamp     = None

        # bag 播放结束检测
        self.last_msg_wall_time = time.time()
        self.bag_started        = False
        self.bag_ended          = False

        # 问题帧记录
        self.issues = []

        self.sub_lidar = self.create_subscription(
            PointCloud2, LIDAR_TOPIC, self.lidar_cb, 10)
        self.sub_imu = self.create_subscription(
            Imu, IMU_TOPIC, self.imu_cb, 100)

        self.summary_timer = self.create_timer(5.0, self.print_summary)
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)

        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}  Bag 质量检测启动{RESET}")
        print(f"{CYAN}  点云: {LIDAR_TOPIC}  (期望 {LIDAR_HZ_EXPECTED} Hz){RESET}")
        print(f"{CYAN}  IMU:  {IMU_TOPIC}  (期望 {IMU_HZ_EXPECTED} Hz){RESET}")
        print(f"{CYAN}  等待 bag 播放...{RESET}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    # ==================== bag 结束检测 ====================
    def check_timeout(self):
        if not self.bag_started or self.bag_ended:
            return
        if time.time() - self.last_msg_wall_time > TIMEOUT_SEC:
            self.bag_ended = True
            print(f"\n{YELLOW}检测到 bag 播放结束（{TIMEOUT_SEC}秒无数据）{RESET}")
            self.print_final_report()
            rclpy.shutdown()

    # ==================== 点云回调 ====================
    def lidar_cb(self, msg):
        self.lidar_count += 1
        self.last_msg_wall_time = time.time()
        self.bag_started = True

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.bag_start_stamp is None:
            self.bag_start_stamp = stamp
        self.bag_end_stamp = stamp

        frame_issues = []

        # 帧间隔检查
        if self.last_lidar_stamp is not None:
            gap = stamp - self.last_lidar_stamp
            if gap > STAMP_GAP_WARN:
                self.lidar_bad_stamp += 1
                frame_issues.append(f"时间戳跳变 gap={gap:.3f}s")
        self.last_lidar_stamp = stamp

        # Hz 统计
        self.lidar_ts_window.append(time.time())
        hz = self._calc_hz(self.lidar_ts_window)
        if hz is not None:
            self.lidar_hz_min = min(self.lidar_hz_min, hz)
            self.lidar_hz_max = max(self.lidar_hz_max, hz)
            if hz < LIDAR_HZ_EXPECTED * HZ_WARN_RATIO:
                self.lidar_hz_warn += 1

        # nan 检查
        total = msg.width * msg.height
        if total == 0:
            self.lidar_bad_points += 1
            frame_issues.append("空帧")
            self._log_issue("LIDAR", stamp, frame_issues)
            return

        points = list(pc2.read_points(
            msg, field_names=["x", "y", "z"], skip_nans=False))
        nan_count = sum(
            1 for p in points
            if not math.isfinite(p[0]) or
               not math.isfinite(p[1]) or
               not math.isfinite(p[2])
        )
        nan_ratio = nan_count / len(points)
        self.lidar_nan_max = max(self.lidar_nan_max, nan_ratio)

        if nan_ratio > NAN_RATIO_WARN:
            self.lidar_bad_nan += 1
            frame_issues.append(f"nan过多 {nan_ratio*100:.1f}%")

        valid_count = len(points) - nan_count
        if valid_count < MIN_POINTS_WARN:
            self.lidar_bad_points += 1
            frame_issues.append(f"有效点不足 {valid_count}pts")

        if frame_issues:
            self._log_issue("LIDAR", stamp, frame_issues)

    # ==================== IMU 回调 ====================
    def imu_cb(self, msg):
        self.imu_count += 1
        self.last_msg_wall_time = time.time()
        self.bag_started = True

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame_issues = []

        # 帧间隔检查
        if self.last_imu_stamp is not None:
            gap = stamp - self.last_imu_stamp
            if gap > IMU_STAMP_GAP_WARN:
                self.imu_bad_stamp += 1
                frame_issues.append(f"时间戳跳变 gap={gap*1000:.1f}ms")
        self.last_imu_stamp = stamp

        # Hz 统计
        self.imu_ts_window.append(time.time())
        hz = self._calc_hz(self.imu_ts_window)
        if hz is not None:
            self.imu_hz_min = min(self.imu_hz_min, hz)
            self.imu_hz_max = max(self.imu_hz_max, hz)
            if hz < IMU_HZ_EXPECTED * HZ_WARN_RATIO:
                self.imu_hz_warn += 1

        # 加速度检查
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        acc_norm = math.sqrt(ax*ax + ay*ay + az*az)
        if acc_norm > IMU_ACC_WARN:
            self.imu_bad_acc += 1
            frame_issues.append(f"加速度异常 {acc_norm:.1f}m/s²")

        # 角速度检查
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        gyr_norm = math.sqrt(gx*gx + gy*gy + gz*gz)
        if gyr_norm > IMU_GYR_WARN:
            self.imu_bad_gyr += 1
            frame_issues.append(f"角速度异常 {gyr_norm:.1f}rad/s")

        if frame_issues:
            self._log_issue("IMU", stamp, frame_issues)

    # ==================== Hz 计算 ====================
    def _calc_hz(self, ts_window):
        if len(ts_window) < 2:
            return None
        duration = ts_window[-1] - ts_window[0]
        if duration <= 0:
            return None
        return (len(ts_window) - 1) / duration

    # ==================== 记录问题 ====================
    def _log_issue(self, source, stamp, issues):
        msg_str = f"[{source}] t={stamp:.3f}s " + " | ".join(issues)
        self.issues.append(msg_str)
        print(f"{RED}⚠  {msg_str}{RESET}")

    # ==================== 实时统计 ====================
    def print_summary(self):
        if self.lidar_count == 0 and self.imu_count == 0:
            print(f"{YELLOW}等待数据... 请确认 bag 已在播放{RESET}")
            return

        lidar_hz_now = self._calc_hz(self.lidar_ts_window)
        imu_hz_now   = self._calc_hz(self.imu_ts_window)

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  实时统计{RESET}")
        print(f"{'='*60}")

        # 点云
        lidar_ok = self.lidar_count - self.lidar_bad_nan \
                   - self.lidar_bad_points - self.lidar_bad_stamp
        lidar_score = lidar_ok / self.lidar_count * 100 if self.lidar_count > 0 else 0
        sc = GREEN if lidar_score > 90 else YELLOW if lidar_score > 70 else RED
        hc = GREEN if (lidar_hz_now or 0) >= LIDAR_HZ_EXPECTED * HZ_WARN_RATIO else RED

        print(f"\n{BOLD}【点云】{RESET}")
        print(f"  总帧数:      {self.lidar_count}")
        if lidar_hz_now:
            print(f"  当前频率:    {hc}{lidar_hz_now:.1f} Hz{RESET}  (期望 {LIDAR_HZ_EXPECTED} Hz)")
        if self.lidar_hz_min != float('inf'):
            print(f"  频率范围:    {self.lidar_hz_min:.1f} ~ {self.lidar_hz_max:.1f} Hz")
        print(f"  Hz不足次数:  {self.lidar_hz_warn}")
        print(f"  nan异常帧:   {self.lidar_bad_nan}")
        print(f"  点数不足帧:  {self.lidar_bad_points}")
        print(f"  时间戳跳变:  {self.lidar_bad_stamp}")
        print(f"  最大nan比例: {self.lidar_nan_max*100:.1f}%")
        print(f"  质量评分:    {sc}{lidar_score:.1f}%{RESET}")

        # IMU
        imu_ok = self.imu_count - self.imu_bad_acc \
                 - self.imu_bad_gyr - self.imu_bad_stamp
        imu_score = imu_ok / self.imu_count * 100 if self.imu_count > 0 else 0
        sc = GREEN if imu_score > 90 else YELLOW if imu_score > 70 else RED
        hc = GREEN if (imu_hz_now or 0) >= IMU_HZ_EXPECTED * HZ_WARN_RATIO else RED

        print(f"\n{BOLD}【IMU】{RESET}")
        print(f"  总帧数:      {self.imu_count}")
        if imu_hz_now:
            print(f"  当前频率:    {hc}{imu_hz_now:.1f} Hz{RESET}  (期望 {IMU_HZ_EXPECTED} Hz)")
        if self.imu_hz_min != float('inf'):
            print(f"  频率范围:    {self.imu_hz_min:.1f} ~ {self.imu_hz_max:.1f} Hz")
        print(f"  Hz不足次数:  {self.imu_hz_warn}")
        print(f"  加速度异常:  {self.imu_bad_acc}")
        print(f"  角速度异常:  {self.imu_bad_gyr}")
        print(f"  时间戳跳变:  {self.imu_bad_stamp}")
        print(f"  质量评分:    {sc}{imu_score:.1f}%{RESET}")
        print(f"\n{'='*60}\n")

    # ==================== 最终报告 ====================
    def print_final_report(self):
        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}  最终质量报告{RESET}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}")

        if self.bag_start_stamp and self.bag_end_stamp:
            duration = self.bag_end_stamp - self.bag_start_stamp
            print(f"\n{BOLD}Bag 时长: {duration:.1f}s{RESET}")

        self.print_summary()

        if not self.issues:
            print(f"{GREEN}✓ 未发现问题帧{RESET}\n")
        else:
            print(f"{BOLD}{RED}发现 {len(self.issues)} 个问题帧（显示前20条）：{RESET}")
            for i, issue in enumerate(self.issues[:20]):
                print(f"  {i+1}. {issue}")
            if len(self.issues) > 20:
                print(f"  ... 共 {len(self.issues)} 个问题")

        # 综合评分
        total_bad = (self.lidar_bad_nan + self.lidar_bad_points +
                     self.lidar_bad_stamp + self.lidar_hz_warn +
                     self.imu_bad_acc + self.imu_bad_gyr +
                     self.imu_bad_stamp + self.imu_hz_warn)
        total = self.lidar_count + self.imu_count
        score = max(0.0, (1 - total_bad / total) * 100) if total > 0 else 0

        color = GREEN if score > 90 else YELLOW if score > 70 else RED
        print(f"\n{BOLD}综合评分: {color}{score:.1f}/100{RESET}")

        if score > 90:
            print(f"{GREEN}✓ 优秀，适合建图{RESET}")
        elif score > 70:
            print(f"{YELLOW}△ 一般，建图可能有局部漂移{RESET}")
        else:
            print(f"{RED}✗ 较差，建议重新录制{RESET}")

        print(f"\n{BOLD}建议：{RESET}")
        if self.lidar_bad_nan > 0:
            print(f"  {YELLOW}· 点云 nan 比例高，避免对着玻璃/镜面录制{RESET}")
        if self.imu_bad_stamp > 0:
            print(f"  {YELLOW}· IMU 时间戳跳变多，录制时关闭其他占 CPU 的程序{RESET}")
        if self.lidar_hz_warn > 0:
            print(f"  {YELLOW}· 点云频率不稳定，检查雷达连接和 CPU 负载{RESET}")
        if self.imu_hz_warn > 0:
            print(f"  {YELLOW}· IMU 频率不稳定，可能存在数据丢包{RESET}")
        if score > 90:
            print(f"  {GREEN}· bag 质量良好，无需特别处理{RESET}")
        print()


def main():
    rclpy.init()
    node = BagQualityChecker()

    def on_exit(sig, frame):
        if not node.bag_ended:
            node.print_final_report()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if not node.bag_ended:
            node.print_final_report()
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == '__main__':
    main()