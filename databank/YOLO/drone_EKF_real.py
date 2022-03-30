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

if __name__ == '__main__':
    
    ekf = DroneEKF()
    
    dados=pd.read_csv('dados_camera_processa.csv')
    x=dados['x'].tolist()
    y=dados['y'].tolist()
    z=dados['z'].tolist()
    centroide_x=dados['meio_x']
    yaw=dados['yaw'].tolist()
    t=dados['tempo'].tolist()
    
    x_k=[]
    y_k=[]
    yaw_k=[]
    yaw_k.append(yaw[0])
    t_k=[]
    
    #noise=np.random.normal(0,0.05,len(yaw))
    yaw_camera=[]
    img_x=600/2
    img_y=555/2
    kp=30
    for j in range(0, len(yaw)):
        res=yaw[j]+kp*abs(centroide_x[j]-img_x)/img_x
        yaw_camera.append(res)
    
    for i in range(1, len(x)):
        Dt=t[i]-t[i-1]
        v=np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)/Dt
        om=(yaw[i]-yaw[i-1])/Dt
        Tk=[x[i], y[i], yaw[i], om, v, Dt]
        dados_kalman=ekf.step(Tk)
        
        x_k.append(dados_kalman[0])
        y_k.append(dados_kalman[1])
        yaw_k.append(dados_kalman[2])
        
    plt.rcParams['figure.dpi'] = 360
    plt.scatter(t[:37], yaw[:37], marker='.', color='blue', label='Yaw_compass')
    plt.scatter(t[:37], yaw_camera[:37], marker='.', color='green', label='Yaw_camera')
    plt.plot(t[:37], yaw_k[:37], color='red', label='Yaw_kalman')
    plt.ylabel('Drone Orientation (º)')
    plt.xlabel('Simulation Time (s)')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(x, y, color='r', label='Trajectory Stage 2')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.legend()
    plt.title('Detailed Trajectory Stage 2')
    plt.grid()
    plt.show()
    
    plt.clf()
    fig2 = plt.figure(figsize=(8,6))
    ax1 = plt.axes(projection='3d')
    plt.plot(x, y, z, color='r', label='Trajectory Stage 2')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)', labelpad=10)
    ax1.set_zlabel('UAV Z position (m)')
    plt.legend()
    plt.title('Detailed Trajectory Stage 2')
    plt.grid()
    ax1.view_init(30, -75)
    plt.show()
    

    