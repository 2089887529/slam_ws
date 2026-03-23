#!/usr/bin/env python3
"""
IMU 单位转换节点
订阅速腾雷达 IMU（加速度单位：g）
发布转换后的 IMU（加速度单位：m/s²）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

GRAVITY = 9.80665

class ImuConverter(Node):
    def __init__(self):
        super().__init__('imu_converter')

        self.declare_parameter('input_topic', '/rslidar_imu_data')
        self.declare_parameter('output_topic', '/rslidar_imu_converted')
        self.declare_parameter('acc_scale', GRAVITY)

        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.acc_scale = self.get_parameter('acc_scale').value

        self.pub = self.create_publisher(Imu, output_topic, 100)
        self.sub = self.create_subscription(Imu, input_topic, self.cb, 100)

        self.get_logger().info(f'IMU converter started')
        self.get_logger().info(f'  input:  {input_topic}')
        self.get_logger().info(f'  output: {output_topic}')
        self.get_logger().info(f'  acc_scale: {self.acc_scale}')

    def cb(self, msg: Imu):
        out = Imu()
        out.header = msg.header

        out.linear_acceleration.x = -msg.linear_acceleration.y * self.acc_scale
        out.linear_acceleration.y = -msg.linear_acceleration.x * self.acc_scale
        out.linear_acceleration.z = -msg.linear_acceleration.z * self.acc_scale

        out.angular_velocity.x = -msg.angular_velocity.y
        out.angular_velocity.y = -msg.angular_velocity.x
        out.angular_velocity.z = -msg.angular_velocity.z

        out.orientation_covariance[0] = -1.0
        out.angular_velocity_covariance = [0.0] * 9
        out.linear_acceleration_covariance = [0.0] * 9

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()