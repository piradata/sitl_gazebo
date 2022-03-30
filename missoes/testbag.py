#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:41:24 2020

@author: gabryelsr
"""

import rosbag
from std_msgs.msg import Int32, String
    
bag = rosbag.Bag('test.bag', 'w')
j=0 
while j<=20:
    wrd = String()
    wrd.data = 'foo'+str(j)
    
    i = Int32()
    i.data = j
 
    bag.write('chatter', str)
    bag.write('numbers', i)
    
    j=j+1
    
bag.close()