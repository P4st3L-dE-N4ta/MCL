#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

# Global flag to capture only the first amcl_pose
got_initial_pose = False

# Pose representa a pose do robô no espaço 2D
class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

def QuartOrient_to_Orient(orientation_q):
    #orientation_q = odom.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    theta = yaw  # This is your 2D orientation
    return theta

def odom_callback(msg):
    pos = msg.pose.pose.position
    theta = QuartOrient_to_Orient(msg.pose.pose.orientation)
    print(f"Odom received: x={pos.x:.2f}, y={pos.y:.2f}, theta={theta:.2f}")
    return Pose(pos.x, pos.y, theta)

def scan_callback(msg):
    #angle_min = msg.angle_min
    #angle_max = msg.angle_max
    #angle_increment = msg.angle_increment
    #time_increment = msg.time_increment
    #range_min = msg.range_min
    #range_max = msg.range_max
    ranges = msg.ranges
    #intensities = msg.intensities
    print(f"Laser Scan ranges received: max={max(ranges):.2f}, min={min(ranges):.2f}")
    return ranges
    
# Recebe apenas a primeira posicao para o initialpose
def amcl_callback(msg):
    global got_initial_pose
    if not got_initial_pose:
        pos = msg.pose.pose.position
        theta = QuartOrient_to_Orient(msg.pose.pose.orientation)
        print(f"Initial Pose - First AMCL pose: x={pos.x:.2f}, y={pos.y:.2f}, theta={theta:.2f}")
        got_initial_pose = True
        return Pose(pos.x, pos.y, theta)
    

def listener():
    rospy.init_node('my_listener')
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, amcl_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()





