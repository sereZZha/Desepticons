#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


LIDAR_OFFSET_X = 0.220  # Смещение base_scan относительно trunk по X (м)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler(x: float, y: float, z: float, w: float):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quat(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12 or not math.isfinite(norm):
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


class LidarImuOdomNode(Node):
    def __init__(self):
        super().__init__('odom_node')

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_cb, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data
        )

        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Состояние IMU
        self.yaw0 = None
        self.raw_yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0

        # Состояние позы и ключевого кадра лидара
        self.initialized = False
        self.scan_pos = np.zeros(2, dtype=np.float64)
        self.cur_yaw = 0.0
        self.prev_yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_scan_time = None

        self.kf_tree = None
        self.kf_pts = None
        self.kf_normals = None
        self.kf_valid_seg = None
        self.kf_scan_pos = np.zeros(2, dtype=np.float64)
        self.kf_yaw = 0.0

        # Публикация 40 Гц симуляционного времени (норма >= 20 Гц)
        self.timer = self.create_timer(0.025, self.publish_odom)
        self.get_logger().info('T4 Point-to-Line LiDAR+IMU Odom node started')

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        if not (math.isfinite(q.x) and math.isfinite(q.y) and math.isfinite(q.z) and math.isfinite(q.w)):
            return
        r, p, y = quat_to_euler(q.x, q.y, q.z, q.w)
        if self.yaw0 is None:
            self.yaw0 = y
        self.roll = r
        self.pitch = p
        self.raw_yaw = wrap_angle(y - self.yaw0)

        av = msg.angular_velocity
        if math.isfinite(av.x) and math.isfinite(av.y) and math.isfinite(av.z):
            self.gyro_x = av.x
            self.gyro_y = av.y
            self.gyro_z = av.z

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > max(msg.range_min, 0.15)) & (ranges < min(msg.range_max, 18.0))
        idxs = np.nonzero(valid)[0]
        if idxs.size < 30:
            return np.empty((0, 2), dtype=np.float64)
        r = ranges[idxs]
        angles = msg.angle_min + idxs * msg.angle_increment
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))

    def set_keyframe(self, pts_rot: np.ndarray, pos: np.ndarray, yaw: float):
        self.kf_pts = pts_rot.copy()
        seg_vec = np.roll(self.kf_pts, -1, axis=0) - self.kf_pts
        seg_len = np.linalg.norm(seg_vec, axis=1)
        self.kf_valid_seg = (seg_len > 1e-4) & (seg_len < 0.30)
        normals = np.zeros_like(self.kf_pts)
        safe_len = np.where(self.kf_valid_seg, seg_len, 1.0)
        ux = seg_vec[:, 0] / safe_len
        uy = seg_vec[:, 1] / safe_len
        normals[:, 0] = -uy
        normals[:, 1] = ux
        self.kf_normals = normals
        self.kf_tree = cKDTree(self.kf_pts)
        self.kf_scan_pos = pos.copy()
        self.kf_yaw = yaw

    def scan_cb(self, msg: LaserScan):
        if self.yaw0 is None:
            return

        pts_local = self.scan_to_points(msg)
        if pts_local.shape[0] < 30:
            return

        yaw_now = self.raw_yaw
        c, s = math.cos(yaw_now), math.sin(yaw_now)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts_rot = pts_local @ rot.T

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.initialized:
            self.set_keyframe(pts_rot, np.zeros(2, dtype=np.float64), yaw_now)
            self.cur_yaw = yaw_now
            self.prev_yaw = yaw_now
            self.last_scan_time = stamp_sec
            self.initialized = True
            return

        # 2D Point-to-Line ICP (минимизация расстояния по нормалям к стенам)
        t = self.scan_pos - self.kf_scan_pos
        for it in range(15):
            shifted = pts_rot + t
            max_dist = 0.45 if it < 5 else 0.18
            dists, idxs = self.kf_tree.query(shifted, distance_upper_bound=max_dist)
            mask = (dists < max_dist) & self.kf_valid_seg[idxs]
            if np.sum(mask) < 20:
                break
            matched_pts = self.kf_pts[idxs[mask]]
            n = self.kf_normals[idxs[mask]]
            err = np.sum((matched_pts - shifted[mask]) * n, axis=1)
            inliers = np.abs(err) < (0.20 if it < 5 else 0.06)
            if np.sum(inliers) < 15:
                break
            n_in = n[inliers]
            err_in = err[inliers]
            H = n_in.T @ n_in + 1e-3 * np.eye(2)
            b = n_in.T @ err_in
            step = np.linalg.solve(H, b)
            t += step
            if np.linalg.norm(step) < 1e-4:
                break

        candidate_pos = self.kf_scan_pos + t
        step_dist = float(np.linalg.norm(candidate_pos - self.scan_pos))
        gyro_norm = math.sqrt(self.gyro_x ** 2 + self.gyro_y ** 2 + self.gyro_z ** 2)

        dt = (stamp_sec - self.last_scan_time) if self.last_scan_time is not None else 0.0
        self.last_scan_time = stamp_sec

        # Антидрейф (ZUPT): блокируем шум только при реальной остановке
        if step_dist < 0.0025 and gyro_norm < 0.025:
            self.vx = 0.0
            self.vy = 0.0
            self.wz = 0.0
            return

        if dt > 1e-4 and step_dist < 0.5:
            world_vx = (candidate_pos[0] - self.scan_pos[0]) / dt
            world_vy = (candidate_pos[1] - self.scan_pos[1]) / dt
            self.vx = c * world_vx + s * world_vy
            self.vy = -s * world_vx + c * world_vy
            self.wz = wrap_angle(yaw_now - self.prev_yaw) / dt

        if step_dist < 0.5:
            self.scan_pos = candidate_pos
        self.prev_yaw = self.cur_yaw
        self.cur_yaw = yaw_now

        # Обновляем опорный скан каждые 0.40 м или 15 градусов
        if (
            np.linalg.norm(self.scan_pos - self.kf_scan_pos) > 0.40
            or abs(wrap_angle(self.cur_yaw - self.kf_yaw)) > 0.25
        ):
            self.set_keyframe(pts_rot, self.scan_pos, self.cur_yaw)

    def publish_odom(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return

        # Перевод позиции из base_scan в центр trunk (вынос 0.220 м)
        c, s = math.cos(self.cur_yaw), math.sin(self.cur_yaw)
        trunk_x = float(self.scan_pos[0] - LIDAR_OFFSET_X * (c - 1.0))
        trunk_y = float(self.scan_pos[1] - LIDAR_OFFSET_X * s)

        qx, qy, qz, qw = euler_to_quat(self.roll, self.pitch, self.cur_yaw)
        stamp = now.to_msg()

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'trunk'
        tf_msg.transform.translation.x = trunk_x
        tf_msg.transform.translation.y = trunk_y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'trunk'
        odom_msg.pose.pose.position.x = trunk_x
        odom_msg.pose.pose.position.y = trunk_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = float(self.vx)
        odom_msg.twist.twist.linear.y = float(self.vy)
        odom_msg.twist.twist.angular.z = float(self.wz)
        self.odom_pub.publish(odom_msg)


def main():
    rclpy.init()
    node = LidarImuOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
