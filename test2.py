#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from math import radians

l = input('please enter:')

class Steering():
    def __init__(self,act):
        self.s = act
        rospy.init_node('steering',anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        r = rospy.Rate(5)
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(30)
        turn_cmd1 = Twist()
        turn_cmd1.linear.x = 0
        turn_cmd1.angular.z = radians(60)
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = -radians(30)
        move_cmd1 = Twist()
        move_cmd1.linear.x = 0
        move_cmd1.angular.z = -radians(60)
        #s=l.index(1)
        while not rospy.is_shutdown():
            if self.s==0:
                for x in range(0,5):
                    self.cmd_vel.publish(turn_cmd1)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==1:
                for x in range(0,5):
                    self.cmd_vel.publish(turn_cmd)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==2:
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==3:
                for x in range(0,5):
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==4:
                for x in range(0,5):
                    self.cmd_vel.publish(move_cmd1)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            else:
                self.cmd_vel.publish(Twist())
                rospy.spin()
    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
if __name__ == '__main__':
    try:
        Steering()
    except:
        rospy.loginfo("node terminated.")
