#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:55:45 2020

@author: gabryelsr
"""

import numpy as np

def corrige_destino(lf, lof, lnavio, apnavio, pos, des):
    R=6.37279*10**6
    if pos=="proa":
        if des=="meianau":
            d=lnavio/2
        elif des=="popa":
            d=lnavio
        apnavio=apnavio+180
    elif pos=="meianau":
        if des=="proa":
            d=lnavio/2
        elif des=="popa":
            d=lnavio/2
            apnavio=apnavio+180
    elif pos=="popa":
        if des=="proa":
            d=lnavio
        elif des=="meianau":
            d=lnavio/2
    elif pos==des:
        d=0
    
    print(apnavio)
    print(d)
    teste=(np.sin(np.radians(lf))*np.cos(d/R)+np.cos(np.radians(lf))*np.sin(d/R)*np.cos(np.radians(apnavio)))
    print(teste)
    lfc=np.arcsin((np.sin(np.radians(lf))*np.cos(d/R)+np.cos(np.radians(lf))*np.sin(d/R)*np.cos(np.radians(apnavio))))
    lfc=lfc*180.0/np.pi
    lofc=lof+np.arctan2(np.sin(np.radians(apnavio))*np.sin(d/R)*np.cos(np.radians(lf)), np.cos(d/R)-np.sin(np.radians(lf))*np.sin(np.radians(lof)))*180/np.pi
    #latB = asin( sin( latA) * cos( d / R ) + cos( latA ) * sin( d / R ) * cos( θ ))
    #lonB = lonA + atan2(sin( θ ) * sin( d / R ) * cos( latA ), cos( d / R ) − sin( latA ) * sin( latB ))
    return(lfc, lofc)

#destino
des="meianau"
#posição da coordenada de destino
pos="proa"
lon_final=-42.9640769
lat_final=-26.2382942
#comprimento e aproamento do navio
lnavio=310
apnavio=10.1

(lfc, lofc)=corrige_destino(lat_final, lon_final, lnavio, apnavio, pos, des)
print(lfc)
print(lofc)