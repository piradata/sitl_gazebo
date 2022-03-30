import rospy
import mavros
import numpy
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import *
from mavros_msgs.srv import *

def cria_waypoints(pontos, altitude, altura):

	# wp.frame:
	#uint8 frame
	#uint8 FRAME_GLOBAL = 0
	#uint8 FRAME_LOCAL_NED = 1
	#uint8 FRAME_MISSION = 2
	#uint8 FRAME_GLOBAL_REL_ALT = 3
	#uint8 FRAME_LOCAL_ENU = 4
	
	# wp.command:
	#uint16 command
	#uint16 NAV_WAYPOINT = 16
	#uint16 NAV_LOITER_UNLIM = 17
	#uint16 NAV_LOITER_TURNS = 18
	#uint16 NAV_LOITER_TIME = 19
	#uint16 NAV_RETURN_TO_LAUNCH = 20
	#uint16 NAV_LAND = 21
	#uint16 NAV_TAKEOFF = 22
    
	limpa_waypoints()
	fr=3
	wl = WaypointList()
	
	wp = Waypoint()
	wp.frame = fr
	wp.command = 22  # decola
	wp.is_current = True
	wp.autocontinue = True
	wp.param1 = altitude  # takeoff altitude
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[1]
	wp.y_long = pontos[0]
	wp.z_alt = altitude
	wl.waypoints.append(wp)
	limpa_waypoints()
	
	wp = Waypoint() 
	wp.frame = fr
	wp.command = 16  #vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[3]
	wp.y_long = pontos[2]
	wp.z_alt = altitude+altura
	wl.waypoints.append(wp)
	limpa_waypoints()

	wp = Waypoint()
	wp.frame = fr
	wp.command = 16  # vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0 
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[5]
	wp.y_long = pontos[4]
	wp.z_alt = altitude+altura
	wl.waypoints.append(wp)
	limpa_waypoints()

	wp = Waypoint()
	wp.frame = fr
	wp.command = 16  # vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0 
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[7]
	wp.y_long = pontos[6]
	wp.z_alt = altitude+altura
	wl.waypoints.append(wp)
	limpa_waypoints()

	print(wl)
	
	try:
	    service = rospy.ServiceProxy('mavros/mission/push', WaypointPush, persistent=True)
	    service(start_index=0, waypoints=wl.waypoints)
	  
	    if service.call(wl.waypoints).success:
	        print 'missao escrita'
	    else:
	        print 'erro ao escrever'
	  
	except rospy.ServiceException, e:
	    print "Service call failed: %s" % e

def limpa_waypoints():
        try:
            response = rospy.ServiceProxy('mavros/mission/clear', WaypointClear)
            return response.call().success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False

msg=None

def callback(data):
	#rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
	global msg
	msg=data.data
	#rospy.loginfo("%s", msg)

rospy.init_node('waypoint_node', anonymous=True)

#def listener():
	# In ROS, nodes are uniquely named. If two nodes with the same
	# name are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	# rospy.init_node('listener', anonymous=True)
	#rospy.Subscriber("configura_missao", String, callback)
	# spin() simply keeps python from exiting until this node is stopped
	# rospy.spin()

if __name__ == '__main__':
	while msg is None:
		#rospy.init_node('waypoint_node', anonymous=True)
		#listener()
		rospy.Subscriber("configura_missao", String, callback)
		if msg is not None:
			dados=msg	
	dados=dados.replace(")","")
	dados=dados.replace("(","")
	dados = [float(item) for item in dados.split(",")]
	alt_inicial=dados[-1]	
	print(alt_inicial)
	h=30
	cria_waypoints(dados, alt_inicial, h)	

