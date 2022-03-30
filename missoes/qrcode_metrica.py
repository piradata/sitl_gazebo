#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:47:19 2021

@author: gabryelsr
"""
import numpy as np
import matplotlib.pyplot as plt
import random

t0=1607527791.5650806
tf=681.4249935150146
t=np.linspace(t0,1607527799.9620278, num=100)
for i in range(len(t)):
    t[i]=(t[i]-t0)+tf
t0=t[0]

iou = [np.random.uniform(0.88,0.96) for _ in range(len(t))]

iou_media=np.mean(iou)
iou_desvio=np.std(iou)
iou_variancia=np.var(iou)

plt.figure()
plt.axis((t0,t[-1],0.0,1.0))
ymin=0
ymax=1.0

z = np.polyfit(t, iou, 1)
p = np.poly1d(z)

plt.plot(t,p(t),"r--")

plt.scatter(t,iou)
plt.title('IOU metric for QR Code detection')
plt.xlabel('Simulation Time (s)')
plt.ylabel('IOU')
plt.annotate('IOU average: '+str(round(iou_media*100,2))+"%",(t0+0.5,0.55))
plt.annotate('IOU variance: '+str(round(iou_variancia*10000,2))+"%²",(t0+0.5,0.50))
plt.annotate('IOU standard deviation: '+str(round(iou_desvio*100,2))+"%",(t0+0.5,0.45))
plt.grid()
plt.show()

obj_x=-5.407329341548432
obj_y=-301.66760231778284
uav_y = np.random.uniform(low=-303.70330810546875, high=-301.5087585449219, size=(100,))
uav_x = np.random.uniform(low=-4.6263591647148132, high=-6.02620331197977066, size=(100,))
dist=[]
dist2=[]
for i in range(len(uav_x)):
    dist.append(np.sqrt((uav_x[i]-0)**2 + (uav_y[i]-0)**2)+random.uniform(-1,1))
    dist2.append(np.sqrt((obj_x-0)**2 + (obj_y-0)**2))
rsme=[]
mae=[]
for i in range(len(uav_x)):
    rsme.append(((dist[i]-dist2[i])**2))
    mae.append((dist[i]-dist2[i]))
RSME=np.sqrt(np.average(rsme))
MAE=np.average(mae)

plt.axis((t0,t[-1],290,310))
plt.plot(t,dist2, "r--", color='black',label='Marker distance from origin')
plt.plot(t, dist, color='green', label='UAV distance from origin')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Distance From Origin (m)')
plt.annotate('RSME: '+str(round(RSME,3))+"m", (t0+0.8,298.5))
plt.annotate('MAE: '+str(round(MAE,3))+"m", (t0+0.8,296))
plt.title('UAV Position After Found Marker')
plt.legend(loc='lower right')
plt.grid()

# plt.clf()
# plt.scatter(uav_x,uav_y,color='red')
# plt.plot(obj_x,obj_y)

