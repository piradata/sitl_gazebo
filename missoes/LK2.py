#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:03:20 2021

@author: gabryelsr
"""

import rospy
from std_msgs.msg import String, Float64

import numpy as np
import sys
#necessario para que o sistema não tente importar o opencv do ROS
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import datetime
import pandas as pd
import mss

sct = mss.mss()

monitor = {"top": 52, "left": 65, "width": 767, "height": 715}
cap = sct.grab(monitor)
#cap=scr.recv()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
old_frame = np.array(cap)
#blur = cv2.GaussianBlur(old_frame,(5,5),0)
#smooth = cv2.addWeighted(blur,1.5,old_frame,-0.5,0)
#old_frame=smooth.copy()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

p1_backup=[]
st_backup=[]
err_backup=[]
p0_backup=[]
t=[]
t.append(0)
ti=datetime.datetime.now()
yaw_lk=[]
yaw_lk.append(180)

rospy.init_node('camera_opf', anonymous=True)
pub = rospy.Publisher('dados_camera_opf', String)#, queue_size=10)
rate = rospy.Rate(10) # 10hz

while(1):
    cap = sct.grab(monitor)
    frame = np.array(cap)
    try:
        #blur = cv2.GaussianBlur(frame,(5,5),0)
        #smooth = cv2.addWeighted(blur,1.3,frame,-0.4,0)
        #frame=smooth.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        break

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        #p1 = p1_backup
        #st = st_backup
        #err = err_backup
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        if p0 is None:
             p0=p0_backup.reshape(-1,1,2)
             p1 = p1_backup
             st = st_backup
             err = err_backup
        else:
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    # print(p1)
    # print(p0)
    # print(st)
    # print(err)
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    tf=datetime.datetime.now()
    dt=((tf-ti).seconds+(tf-ti).microseconds/10e5)
    #print(dt)
    ti=tf
    t.append(t[-1]+dt)
    omega=((a-c)/frame.shape[0])/dt
    nyaw=yaw_lk[-1]+yaw_lk[-1]*omega*dt
    # DeltaT=tf-dataref
    # data.append(str(data1+datetime.timedelta(seconds=DeltaT.seconds, microseconds=DeltaT.microseconds)))
    if abs(nyaw)>=360:
        nyaw=nyaw-int(nyaw/360)*360
    yaw_lk.append(nyaw)
    
    imgr=cv2.resize(img, (frame.shape[1], frame.shape[0]))
    cv2.imshow('frame',imgr)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    old_frame=frame.copy()
    p0_backup=p0
    p0 = good_new.reshape(-1,1,2)
    p1_backup=p1
    st_backup=st
    err_backup=err

cv2.destroyAllWindows()
cap.release()