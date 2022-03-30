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
from mavros_msgs.msg import State
from std_msgs.msg import Float32, Float64, String

#=================Parameter Initializaiton========================
goal_pose = PoseStamped()
current_pose = PoseStamped()
set_velocity = TwistStamped()
current_state = State()


def altitude_hold():
    global goal_pose
    goal_pose.pose.position.z = 2
    
def navegar1(pontos, j=0):
    while j < len(pontos):
        xt=pontos[j][0]
        yt=pontos[j][1]
        zt=pontos[j][2]
        yawt=np.float32(pontos[j][3]*np.pi/180)
        goal_pose.pose.position.x=xt
        goal_pose.pose.position.y=yt
        goal_pose.pose.position.z=zt
        local_position_pub.publish(goal_pose)
        x=current_pose.pose.position.x
        y=current_pose.pose.position.y
        z=current_pose.pose.position.z
        if np.absolute(x-xt)<=2 and np.absolute(y-yt)<=2 and np.absolute(z-zt)<=2:
            j=j+1
            print(j)
        
    

#==============Call Back Functions=====================
def pos_sub_callback(pose_sub_data):
    global current_pose
    current_pose = pose_sub_data

def state_callback(state_data):
    global current_state
    current_state = state_data

msg=None

def callback(data):
	global msg
	msg=data.data

#============Intialize Node, Publishers and Subscribers=================
rospy.init_node('Vel_Control_Node', anonymous = True)
rate = rospy.Rate(10) #publish at 10 Hz
local_position_subscribe = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pos_sub_callback)
local_position_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size = 10)
setpoint_velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size = 10)
state_status_subscribe = rospy.Subscriber('/mavros/state',State,state_callback)
altitude_hold()


#============Define Velocity==============================================
set_velocity.twist.linear.x = 1 #moving 1m/s at x direction

h=30
while msg is None:
    rospy.Subscriber("configura_trajetoria", String, callback)
    if msg is not None:
        dados=msg
        dados=dados.replace("[","")
        dados=dados.replace("]","")
        dados = [float(item) for item in dados.split(",")]
        #pontos=[(30,0,30,0),(30,-212,30,-90),(0,-212,30,-180)]
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
        print(pontos)

        while not rospy.is_shutdown():
            local_position_pub.publish(goal_pose)
        
            if current_state.mode != "OFFBOARD" or not current_state.armed:
                arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
                arm(True)
                set_mode = rospy.ServiceProxy('/mavros/set_mode',mavros_msgs.srv.SetMode)
                mode = set_mode(custom_mode = 'OFFBOARD')
        
                if current_state.mode == "OFFBOARD":
                    rospy.loginfo('Switched to OFFBOARD mode!')
                    
            navegar1(pontos)
            setpoint_velocity_pub.publish(set_velocity)
        
        
            rate.sleep()