#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:25:02 2021

@author: gabryelsr
"""

import serial
from datetime import datetime

try:
    ser = serial.Serial('/dev/ttyACM0', 9600)
except:
    ser = serial.Serial('/dev/ttyACM1', 9600)
    
# trecho=" Posicao do encoder: "
# pulsos=40
# incremento=360/pulsos
ang=[]
inc=0
msg=ser.readline()
#print(msg)
msg=msg.decode("utf-8")
inc=float(msg)
ang.append(inc)
t=[]
t.append(0)
ti=datetime.now()
data=[]
data.append(str(datetime.now()))
while True:
    msg=ser.readline()
    msg=msg.decode("utf-8")
    inc = float(msg) #float(msg[(msg.find(trecho)+len(trecho)):])
    ang.append(inc)
    if abs(ang[-1])>=360:
        ang[-1]=(ang[-1] - int(ang[-1]/360)*360)
    tf=datetime.now()
    dt=((tf-ti).seconds+(tf-ti).microseconds/10e5)
    ti=tf
    t.append(t[-1]+dt)
    data.append(str(datetime.now()))
    print(ang[-1])
    #print(t[-1])
    