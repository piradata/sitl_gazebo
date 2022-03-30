#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:21:03 2020

@author: gabryelsr
"""

import rospy
import numpy as np
import math
import mavros_msgs

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs import srv
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget
from std_msgs.msg import Float32, Float64, String

#=================Parameter Initializaiton========================
goal_pose = PoseStamped()
current_pose = PoseStamped()
set_velocity = TwistStamped()
current_state = State()


def altitude_hold():
    global goal_pose
    goal_pose.pose.position.z = 20

#==============Call Back Functions=====================
def pos_sub_callback(pose_sub_data):
    global current_pose
    current_pose = pose_sub_data

def state_callback(state_data):
    global current_state
    current_state = state_data

def callback(data):
	global msg
	msg=data.data

def position_distance(cur_p, target_p, threshold=0.1):
        delta_x = math.fabs(cur_p.pose.position.x - target_p.position.x)
        delta_y = math.fabs(cur_p.pose.position.y - target_p.position.y)
        delta_z = math.fabs(cur_p.pose.position.z - target_p.position.z)

        if (delta_x + delta_y + delta_z < threshold):
            return True
        else:
            return False


def local_pose_callback(msg):
        local_pose = msg
        local_enu_position = msg


def mavros_state_callback(msg):
        mavros_state = msg.mode


def imu_callback(msg):
        global global_imu, current_heading
        imu = msg
        current_heading = q2yaw(imu.orientation)

        self.received_imu = True


    def gps_callback(self, msg):
        self.gps = msg

    def FLU2ENU(self, msg):

        FLU_x = msg.pose.position.x * math.cos(self.current_heading) - msg.pose.position.y * math.sin(self.current_heading)
        FLU_y = msg.pose.position.x * math.sin(self.current_heading) + msg.pose.position.y * math.cos(self.current_heading)
        FLU_z = msg.pose.position.z

        return FLU_x, FLU_y, FLU_z


    def set_target_position_callback(self, msg):
        print("Received New Position Task!")

        if msg.header.frame_id == 'base_link':
            '''
            BODY_FLU
            '''
            # For Body frame, we will use FLU (Forward, Left and Up)
            #           +Z     +X
            #            ^    ^
            #            |  /
            #            |/
            #  +Y <------body

            self.frame = "BODY"

            print("body FLU frame")

            ENU_X, ENU_Y, ENU_Z = self.FLU2ENU(msg)

            ENU_X = ENU_X + self.local_pose.pose.position.x
            ENU_Y = ENU_Y + self.local_pose.pose.position.y
            ENU_Z = ENU_Z + self.local_pose.pose.position.z

            self.cur_target_pose = self.construct_target(ENU_X,
                                                         ENU_Y,
                                                         ENU_Z,
                                                         self.current_heading)


        else:
            '''
            LOCAL_ENU
            '''
            # For world frame, we will use ENU (EAST, NORTH and UP)
            #     +Z     +Y
            #      ^    ^
            #      |  /
            #      |/
            #    world------> +X

            self.frame = "LOCAL_ENU"
            print("local ENU frame")

            self.cur_target_pose = self.construct_target(msg.pose.position.x,
                                                         msg.pose.position.y,
                                                         msg.pose.position.z,
                                                         self.current_heading)

    '''
     Receive A Custom Activity
     '''

    def custom_activity_callback(self, msg):

        print("Received Custom Activity:", msg.data)

        if msg.data == "LAND":
            print("LANDING!")
            self.state = "LAND"
            self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                         self.local_pose.pose.position.y,
                                                         0.1,
                                                         self.current_heading)

        if msg.data == "HOVER":
            print("HOVERING!")
            self.state = "HOVER"
            self.hover()

        else:
            print("Received Custom Activity:", msg.data, "not supported yet!")


    def set_target_yaw_callback(self, msg):
        print("Received New Yaw Task!")

        yaw_deg = msg.data * math.pi / 180.0
        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     yaw_deg)

    '''
    return yaw from current IMU
    '''
    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad


    def arm(self):
        if self.armService(True):
            return True
        else:
            print("Vehicle arming failed!")
            return False

    def disarm(self):
        if self.armService(False):
            return True
        else:
            print("Vehicle disarming failed!")
            return False


    def offboard(self):
        if self.flightModeService(custom_mode='OFFBOARD'):
            return True
        else:
            print("Vechile Offboard failed")
            return False


    def hover(self):

        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     self.current_heading)

    def takeoff_detection(self):
        if self.local_pose.pose.position.z > 0.1 and self.offboard_state and self.arm_state:
            return True
        else:
            return False

#============Intialize Node, Publishers and Subscribers=================
rospy.init_node('Vel_Control_Node', anonymous = True)
rate = rospy.Rate(10) #publish at 10 Hz
local_position_subscribe = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pos_sub_callback)
local_position_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size = 10)
setpoint_velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size = 10)
local_target_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
state_status_subscribe = rospy.Subscriber('/mavros/state',State,state_callback)
local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, local_pose_callback)
mavros_sub = rospy.Subscriber("/mavros/state", State, mavros_state_callback)
gps_sub = rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_callback)
imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, imu_callback)
set_target_position_sub = rospy.Subscriber("gi/set_pose/position", PoseStamped, set_target_position_callback)
set_target_yaw_sub = rospy.Subscriber("gi/set_pose/orientation", Float32, set_target_yaw_callback)
custom_activity_sub = rospy.Subscriber("gi/set_activity/type", String, custom_activity_callback)
altitude_hold()

#==============Funcoes de Voo=============================================
def construct_target(x, y, z, yaw, yaw_rate = 0.05):
        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = 9

        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z
        
        # target_raw_pose.velocity.x = vx
        # target_raw_pose.velocity.y = vy
        # target_raw_pose.velocity.z = vz
        
        # target_raw_pose.acceleration_or_force.x = afx
        # target_raw_pose.acceleration_or_force.y = afy
        # target_raw_pose.acceleration_or_force.z = afz

        target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        target_raw_pose.yaw = yaw
        target_raw_pose.yaw_rate = yaw_rate

        return target_raw_pose
    
def posicao_atual():
    cur_target_pose = construct_target(local_pose.pose.position.x,
                                       local_pose.pose.position.y,
                                       local_pose.pose.position.z,
                                       current_heading)
    
def navegue(pontos, j=0):
        '''
        main ROS thread
        '''
        while (rospy.is_shutdown() is False):
            local_target_pub.publish(cur_target_pose)
            if j < len(pontos):
            			xt=pontos[j][0]
            			yt=pontos[j][1]
            			zt=pontos[j][2]
            			yawt=np.float32(pontos[j][3]*np.pi/180)
            			self.cur_target_pose = self.construct_target(xt,yt,zt,yawt)
            			self.local_target_pub.publish(self.cur_target_pose)
            			x=self.local_pose.pose.position.x
            			y=self.local_pose.pose.position.y
            			z=self.local_pose.pose.position.z
            			yaw=self.current_heading
            if np.absolute(x-xt)<=2 and np.absolute(y-yt)<=2 and np.absolute(z-zt)<=2:
             			j=j+1
             			print(j)
            		#else:
            			#self.hover()
            		#time.sleep(0.1)	
            		#rospy.spin()

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                	if(self.disarm()):

                    		self.state = "DISARMED"


            time.sleep(0.1)
            print(self.current_heading*180/np.pi)

#============Define Velocity==============================================
set_velocity.twist.linear.x = 1 #moving 1m/s at x direction
h=30

while not rospy.is_shutdown():
    local_position_pub.publish(goal_pose)
    msg=None
    while msg is None:
        rospy.Subscriber("configura_trajetoria", String, callback)
        if msg is not None:
            dados=msg
    dados=dados.replace("[","")
    dados=dados.replace("]","")
    dados = [float(item) for item in dados.split(",")]
    tam=int(len(dados)/3)
    x=np.zeros(tam)
    y=np.zeros(tam)
    yaw=np.zeros(tam)
    for i in range(tam):
        x[i]=dados[i]
        y[i]=dados[i+tam]
        yaw[i]=dados[i+2*tam]
    x=x.tolist()
    y=y.tolist()
    yaw=yaw.tolist()
    pontos=[]
    for i in range(tam):
        pontos.append((x[i],y[i],h,np.float32(yaw[i])))

    if current_state.mode != "OFFBOARD" or not current_state.armed:
        arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        arm(True)
        set_mode = rospy.ServiceProxy('/mavros/set_mode',mavros_msgs.srv.SetMode)
        mode = set_mode(custom_mode = 'OFFBOARD')

        if current_state.mode == "OFFBOARD":
            rospy.loginfo('Switched to OFFBOARD mode!')

    setpoint_velocity_pub.publish(set_velocity)


    rate.sleep()