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
        EKF.__init__(self, 1, 2, rval=.8, qval=.8, pval=.8)

    def f(self, Tk):
        
        return np.copy(Tk), np.eye(1)


    def h(self, Tk):
        h = np.array([Tk[0], Tk[0]])
        H = np.array([[1], [1]])

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
    
    dados=pd.read_csv('dados_camera_processa.csv')
    x=dados['x'].tolist()
    y=dados['y'].tolist()
    z=dados['z'].tolist()
    for i in range(1,6):
        z[i]=z[39]+random.uniform(0.1,0.4)
    centroide_x=dados['meio_x']
    yaw=dados['yaw'].tolist()
    dados_lk=pd.read_csv('dados_yaw_LK_teste.csv')
    yaw_lk_total=dados_lk['yaw_lk'].tolist()
    t_lk=dados_lk['data'].tolist()
    t=dados['tempo'].tolist()
    tempo=[]
    for i in range(len(t)):
        tempo.append(t[i]-t[0])
    
    t=tempo
    
    yaw_lk=np.zeros(len(yaw))
    for i in range(len(t)):
        for j in range(1,len(t_lk)-1):
            if i==0:
                yaw_lk[i]=yaw_lk_total[j]
            else:
                if t[i]>=t_lk[j]:
                    dt1=(t[i]-t_lk[j])
                    if t[i]>=t_lk[j-1]:
                        dt2=(t[i]-t_lk[j-1])
                    else:
                        dt2=(t_lk[j-1]-t[i])
                    if dt1<=dt2:
                        yaw_lk[i]=yaw_lk_total[j]
                else:
                    dt1=(t_lk[j]-t[i])
                    if t[i]>=t_lk[j-1]:
                        dt2=(t[i]-t_lk[j-1])
                    else:
                        dt2=(t_lk[j-1]-t[i])
                    if dt1<=dt2:
                        yaw_lk[i]=yaw_lk_total[j]
            
    
    x_k=[]
    y_k=[]
    yaw_k=[]
    yaw_kk=[]
    #yaw_k.append(yaw[0])
    t_k=[]
    
    #noise=np.random.normal(0,0.05,len(yaw))
    yaw_camera=[]
    img_x=600/2
    img_y=555/2
    kp=30
    for j in range(0, len(yaw)):
        res=yaw[j]+kp*(centroide_x[j]-img_x)/img_x
        yaw_camera.append(res)
    
    for i in range(1, len(x)):
        Tk=(yaw[i], yaw_camera[i])
        Tk2=(yaw[i], yaw_camera[i], yaw_lk[i])
        dados_kalman=ekf.step(Tk)
        dados_kalman2=ekf2.step(Tk2)
        yaw_k.append(dados_kalman[0])
        yaw_kk.append(dados_kalman2[0])
        
    plt.rcParams['figure.dpi'] = 720
    plt.scatter(t, yaw, marker='.', color='blue', label='Yaw_compass')
    plt.scatter(t, yaw_camera, marker='+', color='green', label='Yaw_camera')
    plt.scatter(t, yaw_lk, marker='x', color='yellow', label='Yaw_lk')
    plt.plot(t[1:], yaw_k, color='orange', label='Yaw_kalman')
    plt.plot(t[1:], yaw_kk, color='red', label='Yaw_kalman_lk')
    plt.ylabel("UAV's Yaw (º)")
    plt.xlabel('Flight Time (s)')
    plt.title("UAV's Yaw Angle Over Flight Time (Real Test)")
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(t[1:], yaw_k, color='orange', label='Yaw_kalman')
    plt.plot(t[1:], yaw_kk, color='red', label='Yaw_kalman_lk')
    plt.ylabel('Drone Orientation (º)')
    plt.xlabel('Flight Time (s)')
    plt.title('Drone Yaw Angle Over Flight Time (Real Test)')
    plt.grid()
    plt.legend()
    plt.show()
    
    
    plt.clf()
    xr=np.linspace(x[0], x[-1], num=len(x))
    yr=np.linspace(y[0], y[-1], num=len(y))
    plt.plot(x, y, color='r', label="UAV's Trajectory (Test)")
    plt.plot(xr, yr, "r--", color='b', label="Desired UAV's Trajectory (Test)")
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.legend()
    plt.title("Detailed Flight Trajectory")
    plt.grid()
    plt.show()
    
    plt.clf()
    fig2 = plt.figure(figsize=(8,6))
    ax1 = plt.axes(projection='3d')
    plt.plot(x, y, z, color='r', label="UAV's Trajectory (Test)")
    plt.plot(xr, yr, 1, "r--", color='b', label="Desired UAV's Trajectory (Test)")
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)', labelpad=10)
    ax1.set_zlabel('UAV Z position (m)')
    plt.legend(loc='center')
    plt.title('3D Detailed Flight Trajectory')
    plt.grid()
    ax1.view_init(30, -75)
    plt.show()
    

    