import os
import sys
import termios
import tty
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class SimpleMover(Node):
    def __init__(self):
        super().__init__('simple_mover_keyboard')
        self.pub = self.create_publisher(Twist, '/spot/cmd_vel', 10)
        self.sub = self.create_subscription(
            CompressedImage,
            '/spot/camera/frontright/image/compressed',
            self.image_callback,
            10
        )

        self.image_dir = os.path.join(os.getcwd(), 'agent_movement_images')
        os.makedirs(self.image_dir, exist_ok=True)
        self.img_count = 0
        self.get_logger().info("WASD for movement, Q,E to rotate, x to exit.")

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                # filename = os.path.join(self.image_dir, f"frame_{self.img_count:06d}.png")
                # cv2.imwrite(filename, img)
                filename = os.path.join(self.image_dir, f"current.png")
                cv2.imwrite(filename, img)
                self.img_count += 1
        except Exception as e:
            self.get_logger().error(f"Image save failed: {e}")

    def move(self, lx=0.0, ly=0.0, az=0.0):
        twist = Twist()
        twist.linear.x = lx
        twist.linear.y = -ly
        twist.angular.z = az
        self.pub.publish(twist)

    def stop(self):
        self.move(0.0, 0.0, 0.0)


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    rclpy.init()
    node = SimpleMover()
    speed = 0.1
    turn_speed = 0.2

    try:
        while rclpy.ok():
            key = get_key()
            if key == 'w':
                node.move(lx=speed)
            elif key == 's':
                node.move(lx=-speed)
            elif key == 'a':
                node.move(ly=-speed)
            elif key == 'd':
                node.move(ly=speed)
            elif key == 'q':
                node.move(az=turn_speed)
            elif key == 'e':
                node.move(az=-turn_speed)
            elif key == 'b':
                node.move(lx=0.0, ly=0.0, az=0.0)
            elif key == 'x':
                node.stop()
                break
            else:
                node.stop()

            rclpy.spin_once(node, timeout_sec=0.01)

    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
