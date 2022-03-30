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
        
        x_k.append(dados_kalman[0]+np.random.uniform(low=-0.2, high=0.1))
        y_k.append(dados_kalman[1]+np.random.uniform(low=-0.2, high=0.1))
        yaw_k.append(dados_kalman[2])
    
    dados1=pd.read_csv('log_etapa1.csv')
    x1=dados1['X'].tolist()
    y1=dados1['Y'].tolist()
    z1=dados1['Altura'].tolist()
    t1=dados1['Tempo'].tolist()
    t1_0=t1[0]
    for i in range(len(t1)):
        t1[i]=t1[i]-t1_0
    for i in range(len(t)):
        t[i]=t[i]-t1_0
    yaw1=dados1['Orientacao'].tolist()
    rota=pd.read_csv('rota_calculada_etapa1.csv')
    xr=rota['X'].tolist()
    yr=rota['Y'].tolist()
    
    plt.rcParams['figure.dpi'] = 360
    plt.scatter(t[5:len(t)], yaw[5:len(yaw)], marker='.', color='blue', label='Yaw_compass')
    plt.scatter(t[5:len(t)], yaw_n[5:len(yaw_n)], marker='.', color='green', label='Yaw_camera')
    plt.plot(t[5:len(t)], yaw_k[5:len(yaw_k)], color='red', label='Yaw_kalman')
    plt.ylabel('UAV Orientation (º)')
    plt.xlabel('Simulation Time (s)')
    plt.title('EKF Yaw Prediction')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(xr, yr, color='b', label='Planned trajectory')
    plt.plot(x1, y1,  color='r', label='Performed trajectory')
    plt.xlabel('UAV X position from takeoff (m)')
    plt.ylabel('UAV Y position from takeoff (m)')
    plt.title('Detailed Stage 1')
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
    plt.title('3D Detailed Stage 1')
    plt.legend(loc='center left')
    ax2.view_init(30, -75)
    plt.show()
    
    plt.clf()
    plt.plot(t1, z1, color='b')
    plt.xlabel('Elapsed simulation time (s)')
    plt.ylabel('Flight height (m)')
    plt.title('Flight Height x Simulation Time on Stage 1')
    plt.grid()
    plt.show()
    
    plt.clf()
    plt.plot(t1, yaw1, color='b')
    plt.xlabel('Elapsed simulation time (s)')
    plt.ylabel('UAV orientation (º)')
    plt.title('Yaw x Simulation Time on Stage 1')
    plt.grid()
    plt.show()
    
    plt.clf()
    plt.plot(x_k[6:], y_k[6:], color='b', label='Trajectory Stage 2')
    plt.plot(x, y, color='r', label='Predicted Trajectory Stage 2')
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
    plt.title('3D Detailed Trajectory Stage 2')
    plt.grid()
    ax1.view_init(30, -75)
    plt.show()
    
    plt.clf()
    plt.plot(x1, y1,  color='b', label='Trajectory Stage 1')
    plt.plot(x, y, color='r', label='Trajectory Stage 2')
    plt.xlabel('UAV X position (m)')
    plt.ylabel('UAV Y position (m)')
    plt.title('UAV Route')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    ax.plot3D(x1, y1,  z1, color='b', label='Trajectory Stage 1')
    plt.plot(x, y, z, color='r', label='Trajectory Stage 2')
    ax.set_xlabel('UAV X position (m)')
    ax.set_ylabel('UAV Y position (m)', labelpad=10)
    ax.set_zlabel('UAV Z position (m)')
    ax.set_title('3D UAV Route')
    plt.legend(loc='center left')
    plt.grid()
    ax.view_init(30, -75)
    fig
    

    