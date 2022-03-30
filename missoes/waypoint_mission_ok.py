#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import *
from mavros_msgs.srv import *
import numpy as np


def create_waypoint():
    
	waypoint_clear_client()
	wl = []
	wp = Waypoint()

	wp.frame = 3
	wp.command = 22  # takeoff
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 477.848685669  # takeoff altitude
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = -26.2364021
	wp.y_long = -42.9637599
	wp.z_alt = 477.848685669
	wl.append(wp)
    
	
	wp = Waypoint() 

	wp.frame = 3
	wp.command = 16  #Navigate to waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  # delay 
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = -26.2363620169
	wp.y_long = -42.9640572604
	wp.z_alt = 507.848685669
	wl.append(wp)

    
	wp = Waypoint()

	wp.frame = 3
	wp.command = 16  # Navigate to waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  # delay
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = -26.2382541163
	wp.y_long = -42.9643742629
	wp.z_alt = 507.848685669
	wl.append(wp)

	wp = Waypoint()

	wp.frame = 3
	wp.command = 16 # takeoff
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  # takeoff altitude
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = -26.2382942
	wp.y_long = -42.9640769
	wp.z_alt = 507.848685669
	wl.append(wp)

	print(wl)
	
	try:
	    service = rospy.ServiceProxy('mavros/mission/push', WaypointPush)
	    (service(start_index=000, waypoints=wl))
	    

	  
	    if service.call(wl).success: #burasi belki list istiyordur. birde oyle dene
	        print 'write mission success'
	    else:
	        print 'write mission error'
	  
	except rospy.ServiceException, e:
	    print "Service call failed: %s" % e


	



def waypoint_clear_client():
        try:
            response = rospy.ServiceProxy('mavros/mission/clear', WaypointClear)
            return response.call().success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False


if __name__ == '__main__':
	rospy.init_node('waypoint_node', anonymous=True)
	pub = rospy.Publisher('global',String,queue_size=10)
	autor = "Gabryel"
	pub.publish(autor)
	create_waypoint()

