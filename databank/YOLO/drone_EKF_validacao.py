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
import datetime

class DroneEKF(EKF):
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
    
class DroneEKF2(EKF):
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


if __name__ == '__main__':
    
    ekf = DroneEKF()
    ekf2 = DroneEKF2()
    
    dados=pd.read_csv('dados_camera_yaw_e_yoloff.csv')
    dados_lk=pd.read_csv('dados_yaw_LK.csv')

    # x=dados['x'].tolist()
    # y=dados['y'].tolist()
    # z=dados['z'].tolist()
    centroide_x=dados['meio_x'].tolist()
    img_x=dados['img_x'].tolist()
    yaw=dados['yaw'].tolist()
    data=dados['data'].tolist()
    t=[]
    for i in range(len(data)):
        data[i]=datetime.datetime.strptime(data[i], '%Y-%m-%d %H:%M:%S.%f')
        if i==0:
            t.append(0)
        else:
            t.append((data[i]-data[0]).seconds+(data[i]-data[0]).microseconds/10e5)
    
    yaw_lk_total=dados_lk['yaw_lk'].tolist()
    
    data_lk=dados_lk['data'].tolist()
    for i in range(len(data_lk)):
        data_lk[i]=datetime.datetime.strptime(data_lk[i], '%Y-%m-%d %H:%M:%S.%f')
    
    tref=t[0]
    
    for i in range(len(t)):
        t[i]=t[i]-tref
    
    yaw_lk=np.zeros(len(t))
    
    for i in range(len(data)):
        for j in range(1,len(data_lk)-1):
            if i==0:
                yaw_lk[i]=yaw_lk_total[j]
            else:
                if data[i]>=data_lk[j]:
                    dt1=(data[i]-data_lk[j]).seconds+(data[i]-data_lk[j]).microseconds/10e5
                    if data[i]>=data_lk[j-1]:
                        dt2=(data[i]-data_lk[j-1]).seconds+(data[i]-data_lk[j-1]).microseconds/10e5
                    else:
                        dt2=(data_lk[j-1]-data[i]).seconds+(data_lk[j-1]-data[i]).microseconds/10e5
                    if dt1<=dt2:
                        yaw_lk[i]=yaw_lk_total[j]
                else:
                    dt1=(data_lk[j]-data[i]).seconds+(data_lk[j]-data[i]).microseconds/10e5
                    if data[i]>=data_lk[j-1]:
                        dt2=(data[i]-data_lk[j-1]).seconds+(data[i]-data_lk[j-1]).microseconds/10e5
                    else:
                        dt2=(data_lk[j-1]-data[i]).seconds+(data_lk[j-1]-data[i]).microseconds/10e5
                    if dt1<=dt2:
                        yaw_lk[i]=yaw_lk_total[j]
                    
    x_k=[]
    y_k=[]
    yaw_kk=[]
    yaw_k=[]
    #yaw_k.append(yaw[0])
    t_k=[]
    
    noise=np.random.normal(0,3,len(yaw))
    yaw_camera=[]
    yaw_camera.append(33)
    kp=30
    for j in range(1, len(yaw)):
        res=yaw[j-1]+yaw[j-1]*(centroide_x[j]-centroide_x[j-1])/img_x[j]
        yaw_camera.append(res+noise[j])
        yaw_lk[j]=yaw_lk[j]+noise[j]
    
    for i in range(1, len(yaw)):
        Tk=(yaw[i], yaw_camera[i], yaw_lk[i])
        Tk2=(yaw[i], yaw_camera[i])
        dados_kalman=ekf.step(Tk)
        dados_kalman2=ekf2.step(Tk2)
        yaw_kk.append(dados_kalman[0])
        yaw_k.append(dados_kalman2[0])
    
    dados_enc=pd.read_csv('yaw_encoder_validacao_final.csv')
    yaw_enc=dados_enc['yaw'].tolist()
    
    plt.rcParams['figure.dpi'] = 720
    plt.scatter(t, yaw, marker='.', color='blue', label='Yaw_compass')
    plt.scatter(t, yaw_camera, marker='+', color='green', label='Yaw_camera')
    plt.scatter(t, yaw_lk, marker='x', color='yellow', label='Yaw_lucas-kanade')
    plt.plot(t[1:], yaw_kk, color='red', label='Yaw_kalman_kanade')
    plt.plot(t[1:], yaw_k, color='orange', label='Yaw_kalman')
    plt.plot(t, yaw_enc, color='pink', label='Yaw_encoder')
    plt.ylabel("UAV's Yaw (º)")
    plt.xlabel('Test Time (s)')
    plt.title("UAV's Yaw Angle Over Test Time")
    plt.grid()
    plt.legend()
    plt.show()
    
    mspe_kk=[]
    meas_kk=[]
    mspe_k=[]
    meas_k=[]
    mse_c=[]
    meas_c=[]
    mse_yoloff=[]
    meas_yoloff=[]
    mse_lk=[]
    meas_lk=[]
    
    error=[]
    std=np.std(yaw_kk)
    for i in range(len(yaw_kk)):
        error.append((yaw_kk[i]-yaw_enc[i]))
    
    for i in range(1,len(yaw)):
        mspe_k.append((yaw_enc[i]-yaw_k[i-1])**2)
        meas_k.append((yaw_enc[i]-yaw_k[i-1]))
        mspe_kk.append((yaw_enc[i]-yaw_kk[i-1])**2)
        meas_kk.append((yaw_enc[i]-yaw_kk[i-1]))
        mse_c.append((yaw_enc[i]-yaw[i])**2)
        meas_c.append((yaw_enc[i]-yaw[i]))
        mse_yoloff.append((yaw_enc[i]-yaw_camera[i])**2)
        meas_yoloff.append((yaw_enc[i]-yaw_camera[i]))
        mse_lk.append((yaw_enc[i]-yaw_lk[i])**2)
        meas_lk.append((yaw_enc[i]-yaw_lk[i]))
        
    mspe_k=np.average(mspe_k)
    meas_k=np.average(meas_k)
    mspe_kk=np.average(mspe_kk)
    meas_kk=np.average(meas_kk)
    mse_c=np.average(mse_c)
    meas_c=np.average(meas_c)
    mse_yoloff=np.average(mse_yoloff)
    meas_yoloff=np.average(meas_yoloff)
    mse_lk=np.average(mse_lk)
    meas_lk=np.average(meas_lk)
    
    # plt.clf()
    # plt.plot(x, y, color='r', label='Trajectory Stage 2')
    # plt.xlabel('UAV X position (m)')
    # plt.ylabel('UAV Y position (m)')
    # plt.legend()
    # plt.title('Detailed Flight Trajectory')
    # plt.grid()
    # plt.show()
    
    # plt.clf()
    # fig2 = plt.figure(figsize=(8,6))
    # ax1 = plt.axes(projection='3d')
    # plt.plot(x, y, z, color='r', label='Trajectory Stage 2')
    # plt.xlabel('UAV X position (m)')
    # plt.ylabel('UAV Y position (m)', labelpad=10)
    # ax1.set_zlabel('UAV Z position (m)')
    # plt.legend()
    # plt.title('Detailed Flight Trajectory')
    # plt.grid()
    # ax1.view_init(30, -75)
    # plt.show()
    
    measurement=np.linspace(1, len(error), len(error))
    pstd=np.ones(len(error))*std
    nstd=np.ones(len(error))*-1*std
    
    plt.figure()
    plt.plot(measurement,pstd,color='green', label='σ')
    plt.plot(measurement,nstd,color='red', label='-σ')
    plt.scatter(measurement,error,color='blue', label='EKF Error')
    plt.xlabel('Measurement number')
    plt.title('EKF Standard Deviation (σ) x EKF Error (compared to GT, in degrees)')
    plt.legend()
    plt.grid()

    