#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:47:08 2020

@author: gabryelsr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tinyekf import EKF
from mpl_toolkits import mplot3d
import random

class DroneEKF(EKF):
    '''
    EKF para auxiliar navegação do drone
    '''

    def __init__(self):

        # 6 estados (x, y, yaw, omega, v, tempo), 6 medições, alto erro
        EKF.__init__(self, 6, 6, rval=.8, qval=.8, pval=.8)

    def f(self, Tk):
        
        F=np.eye(len(Tk))
        Dtk=Tk[5]
        yawk=Tk[2]
        vk=Tk[4]
        Omk=Tk[3]
        
        F[0][2]=-Dtk*vk*np.sin(np.radians(yawk))
        F[0][4]=Dtk*np.cos(np.radians(yawk))
        F[0][5]=vk*np.cos(np.radians(yawk))
        
        F[1][2]=Dtk*vk*np.cos(np.radians(yawk))
        F[1][4]=Dtk*np.sin(np.radians(yawk))
        F[1][5]=vk*np.sin(np.radians(yawk))
        
        F[2][3]=Dtk
        F[2][5]=Omk
        
        return np.copy(Tk), F


    def h(self, Tk):

        h=np.copy(Tk)
        H=np.eye(len(Tk))

        return h, H

class DroneEKF2(EKF):
    '''
    EKF para auxiliar navegação do drone
    '''

    def __init__(self):

        # 6 estados (x, y, yaw, omega, v, tempo), 6 medições, alto erro
        EKF.__init__(self, 1, 3, rval=.8, qval=.8, pval=.8)

    def f(self, Tk):
        
        return np.copy(Tk), np.eye(1)


    def h(self, Tk):
        h = np.array([Tk[0], Tk[0], Tk[0]])
        H = np.array([[1], [1], [1]])

        return h, H

if __name__ == '__main__':
    
    ekf = DroneEKF()
    ekf2 = DroneEKF2()
    
    dados=pd.read_csv('log_etapa2.csv')
    x=dados['X'].tolist()
    y=dados['Y'].tolist()
    z=dados['Altura'].tolist()
    yaw=dados['Orientacao'].tolist()
    t=dados['Tempo'].tolist()
    
    x_k=[]
    y_k=[]
    yaw_k=[]
    yaw_k.append(yaw[0])
    t_k=[]
    
    yaw_kk=[]
    yaw_kk.append(yaw[0])
    yaw_camera=[]
    yaw_lk=[]
    for i in range(len(yaw)):
        yaw_camera.append(yaw[i]+random.uniform(-1,1))
        yaw_lk.append(yaw[i]+random.uniform(-2,2))
    
    noise=np.random.normal(0,0.05,len(yaw))
    yaw_n=[]
    for j in range(0, len(yaw)):
        res=yaw[j]+noise[j]
        yaw_n.append(res)
    
    for i in range(1, len(x)):
        Dt=t[i]-t[i-1]
        v=np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)/Dt
        om=(yaw[i]-yaw[i-1])/Dt
        Tk=[x[i], y[i], yaw[i], om, v, Dt]
        dados_kalman=ekf.step(Tk)
        
        Tk2=(yaw[i], yaw_camera[i], yaw_lk[i])
        dados_kalman2=ekf2.step(Tk2)
        yaw_kk.append(dados_kalman2[0])
        
        x_k.append(dados_kalman[0])
        y_k.append(dados_kalman[1])
        yaw_k.append(dados_kalman[2])
        
    plt.rcParams['figure.dpi'] = 360
    plt.scatter(t[5:len(t)], yaw[5:len(yaw)], marker='.', color='blue', label='Yaw_compass')
    plt.scatter(t[5:len(t)], yaw_n[5:len(yaw_n)], marker='.', color='green', label='Yaw_camera')
    plt.plot(t[5:len(t)], yaw_k[5:len(yaw_k)], color='red', label='Yaw_kalman')
    plt.ylabel('Drone Orientation (º)')
    plt.xlabel('Simulation Time (s)')
    plt.grid()
    plt.legend()
    plt.show()
    
    dados1=pd.read_csv('log_etapa1.csv')
    x1=dados1['X'].tolist()
    y1=dados1['Y'].tolist()
    z1=dados1['Altura'].tolist()
    t1=dados1['Tempo'].tolist()
    yaw1=dados1['Orientacao'].tolist()
    rota=pd.read_csv('rota_calculada_etapa1.csv')
    xr=rota['X'].tolist()
    yr=rota['Y'].tolist()
    
    plt.clf()
    plt.plot(xr, yr, color='b', label='Planned trajectory')
    plt.plot(x1, y1,  color='r', label='Performed trajectory')
    plt.xlabel('Drone X position from takeoff (m)')
    plt.ylabel('Drone Y position from takeoff (m)')
    plt.grid()
    plt.title('Detailed Stage 1')
    plt.legend()
    plt.show()
    
    plt.clf()
    fig3 = plt.figure(figsize=(8,6))
    ax2 = plt.axes(projection='3d')
    plt.plot(xr, yr, 5, color='b', label='Planned trajectory')
    plt.plot(x1, y1,  z1, color='r', label='Performed trajectory')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)', labelpad=10)
    ax2.set_zlabel('UAV Z position (m)')
    plt.grid()
    plt.title('Detailed Stage 1')
    plt.legend(loc='center left')
    ax2.view_init(30, -75)
    plt.show()
    
    plt.clf()
    plt.plot(t1, z1, color='b')
    plt.xlabel('Elapsed simulation time (s)')
    plt.ylabel('Flight height (m)')
    plt.grid()
    plt.show()
    
    plt.clf()
    plt.plot(t1, yaw1, color='b')
    plt.xlabel('Elapsed simulation time (s)')
    plt.ylabel('Drone orientation (º)')
    plt.grid()
    plt.show()
    
    plt.clf()
    plt.plot(x, y, color='purple', label="UAV's Trajectory (Simulation)")
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.legend()
    plt.title("UAV's Position on Simulation")
    plt.grid()
    plt.show()
    
    plt.clf()
    fig2 = plt.figure(figsize=(8,6))
    ax1 = plt.axes(projection='3d')
    plt.plot(x, y, z, color='purple', label="3D UAV's Trajectory (Simulation)")
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)', labelpad=10)
    ax1.set_zlabel('UAV Z position (m)')
    plt.legend(loc='center')
    plt.title("3D UAV's Position on Simulation")
    plt.grid()
    ax1.view_init(30, -75)
    plt.show()
    
    plt.clf()
    obj_x=-5.407329341548432
    obj_y=-301.66760231778284
    x2=[xr[-1], obj_x]
    y2=[yr[-1], obj_y]
    z2=[-35, -35]
    plt.plot(x1, y1,  color='green', label='Autonomous trajectory')
    plt.plot(xr, yr, "r--", color='black', label='Planned autonomous trajectory')
    plt.plot(x, y, color='orange', label='Approach trajectory')
    plt.plot(x2, y2, "r--", color='purple', label='Ideal approach trajectory')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.title('UAV Route')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    ax.plot3D(x1, y1,  z1, color='green', label='Autonomous trajectory')
    plt.plot(xr, yr, 5, "r--", color='black', label='Planned autonomous trajectory')
    plt.plot(x, y, z, color='orange', label='Approach trajectory')
    plt.plot(x2, y2, -35, "r--", color='purple', label='Ideal approach trajectory')
    ax.set_xlabel('UAV X position (m)')
    ax.set_ylabel('UAV Y position (m)', labelpad=10)
    ax.set_zlabel('UAV Z position (m)')
    ax.set_title('UAV Route')
    plt.legend(loc='center left')
    plt.grid()
    ax.view_init(30, -75)
    #fig
    
    tn=[]
    t0=t1[0]
    for i in range(len(t)):
        tn.append(t[i]-t0)
        
    for i in range(len(yaw)):
        yaw[i]=yaw[i]+random.uniform(-0.5,0.5)
    
    plt.clf()
    plt.rcParams['figure.dpi'] = 360
    plt.scatter(tn[1:], yaw_camera[1:], marker='x', color='green', label='Yaw_camera')
    plt.scatter(tn[1:], yaw_lk[1:], marker='+', color='yellow', label='Yaw_lk')
    plt.scatter(tn[1:], yaw[1:], marker='.', color='cyan', label='Yaw_compass')
    plt.plot(tn[3:], yaw_kk[3:], color='red', label='Yaw_kalman')
    plt.plot(tn[3:], yaw_k[3:], color='blue', label='Yaw_kalman_lk')
    plt.ylabel("UAV's Yaw (º)")
    plt.xlabel('Flight Time (s)')
    plt.title("UAV's Yaw Angle Over Flight Time (Simulation)")
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.clf()
    obj_x=-5.407329341548432
    obj_y=-301.66760231778284
    x2=[xr[-1], obj_x]
    y2=[yr[-1], obj_y]
    z2=[-35, -35]
    x3=np.random.uniform(x[-1]-1, x[-1]+1, 10)
    y3=np.random.uniform(y[-1]-20, y[-1]+20, 10)
    x4=[x3[-1], x1[0]]
    y4=[y3[-1], y1[0]]
    plt.plot(x1, y1,  color='green', label='Autonomous trajectory')
    plt.plot(x, y, color='orange', label='Approach trajectory')
    plt.plot(x3, y3, color='red', label='Trying to locate marker')
    plt.plot(x4, y4, color='blue', label='Fail safe - Return to home')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.title('UAV Route')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    

    