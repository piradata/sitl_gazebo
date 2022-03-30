import numpy as np
import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String
import time
from pyquaternion import Quaternion
import math
import threading
from mavros_msgs.msg import *
from mavros_msgs.srv import *

class GPS():
	# Iniciando a classe GPS 
	def state_callback(self,data):
		self.cur_state = data
		# Obtem estado de /mavros/state
	def pose_sub_callback(self,pose_sub_data):
		self.current_pose = pose_sub_data
		# Funcao para obter posicaoo atual da FCU
	def gps_callback(self,data):
		self.gps = data
		self.gps_read = True
		# Obtem dados de GPS e seta valor para leitura
	def le_gps(self):
		rospy.init_node('leitura_gps', anonymous=True)	# Inicia no de leitura 
		self.gps_read = False # Na inicializacao a leitura eh setada como falsa
		r = rospy.Rate(10) # Frequencia de comunicacao em 10Hz
		rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback) # Dados de GPS
		self.localtarget_received = False 
		r = rospy.Rate(10)
		rospy.Subscriber("/mavros/state", State, self.state_callback)
		rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback) 
		while not self.gps_read:
			r.sleep()
		latitude = self.gps.latitude #Informacao de latitude
		longitude = self.gps.longitude #Informacao de longitude
		return(latitude,longitude)

def calcula_coordenadas(d,tetag,lir,loir,dirx,diry,R):
	if (dirx=="O" and diry=="N"):
		lat1=np.arcsin(np.sin(lir)*np.cos(d/R)+np.cos(lir)*np.sin(d/R)*np.cos(2*np.pi-tetag*np.pi/180))
		lat1g=lat1*180/np.pi
		lon1=loir+np.arctan2(np.sin(2*np.pi-tetag*np.pi/180)*np.sin(d/R)*np.cos(lir), np.cos(d/R)-np.sin(lir)*np.sin(lat1))
		lon1g=lon1*180/np.pi
	if (dirx=="O" and diry=="S"):
		lat1=np.arcsin(np.sin(lir)*np.cos(d/R)+np.cos(lir)*np.sin(d/R)*np.cos(1.5*np.pi-tetag*np.pi/180))
		lat1g=lat1*180/np.pi
		lon1=loir+np.arctan2(np.sin(1.5*np.pi-tetag*np.pi/180)*np.sin(d/R)*np.cos(lir), np.cos(d/R)-np.sin(lir)*np.sin(lat1))
		lon1g=lon1*180/np.pi
	if (dirx=="L" and diry=="S"):
		lat1=np.arcsin(np.sin(lir)*np.cos(d/R)+np.cos(lir)*np.sin(d/R)*np.cos(np.pi-tetag*np.pi/180))
		lat1g=lat1*180/np.pi
		lon1=loir+np.arctan2(np.sin(np.pi-tetag*np.pi/180)*np.sin(d/R)*np.cos(lir), np.cos(d/R)-np.sin(lir)*np.sin(lat1))
		lon1g=lon1*180/np.pi
	if (dirx=="L" and diry=="N"):
		lat1=np.arcsin(np.sin(lir)*np.cos(d/R)+np.cos(lir)*np.sin(d/R)*np.cos(tetag*np.pi/180))
		lat1g=lat1*180/np.pi
		lon1=loir+np.arctan2(np.sin(tetag*np.pi/180)*np.sin(d/R)*np.cos(lir), np.cos(d/R)-np.sin(lir)*np.sin(lat1))
		lon1g=lon1*180/np.pi
	return(lat1g,lon1g)

def calcula_pontos(lir,loir,lfr,lofr,d):
#	lat_inicio=-42.9637599
#	lon_inicio=-26.2364022
#	lat_final=-42.9640769
#	lon_final=-26.2382942
	R=6.37279*10**6
	lir=np.pi*lir/180
	loir=np.pi*loir/180
	lfr=np.pi*lfr/180
	lofr=np.pi*lofr/180

	Dgama=np.log(np.tan(lfr/2+np.pi/4)/np.tan(lir/2+np.pi/4))
	Dlon=np.absolute(loir-lofr)

	teta=np.arctan2(Dlon,Dgama)
	tetag=teta*180/np.pi
	if tetag>=90 and tetag<180:
		tetag=tetag-90
	elif tetag>=180 and tetag<270:
		tetag=tetag-180
	elif tetag>=270 and tetag<360:
		tetag=tetag-270
	L=R*np.arccos(np.sin(lir)*np.sin(lfr)+np.cos(lir)*np.cos(lfr)*np.cos(loir-lofr))
	(lat1,lon1)=calcula_coordenadas(d,tetag,lir,loir,"O","N",R)
	(lat2,lon2)=calcula_coordenadas(L,tetag,lat1*np.pi/180,lon1*np.pi/180,"O","S",R)
	return (L, tetag, lon1, lat1, lon2, lat2)

def cria_waypoints(pontos,altura):
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
	wl = []
	wp = Waypoint()

	wp.frame = 3
	wp.command = 22  # decola
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  # takeoff altitude
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[0][1]
	wp.y_long = pontos[0][0]
	wp.z_alt = altura
	wl.append(wp)
	
	wp = Waypoint() 

	wp.frame = 3
	wp.command = 16  #vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0  
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[1][1]
	wp.y_long = pontos[1][0]
	wp.z_alt = altura
	wl.append(wp)

	wp = Waypoint()

	wp.frame = 3
	wp.command = 16  # vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0 
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[2][1]
	wp.y_long = pontos[2][0]
	wp.z_alt = altura
	wl.append(wp)

	wp = Waypoint()

	wp.frame = 3
	wp.command = 16  # vai para o waypoint.
	wp.is_current = False
	wp.autocontinue = True
	wp.param1 = 0 
	wp.param2 = 0
	wp.param3 = 0
	wp.param4 = 0
	wp.x_lat = pontos[3][1]
	wp.y_long = pontos[3][0]
	wp.z_alt = altura
	wl.append(wp)

	print(wl)
	
	try:
	    service = rospy.ServiceProxy('mavros/mission/push', WaypointPush, persistent=True)
	    service(start_index=0, waypoints=wl)
	  
	    if service.call(wl).success:
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

if __name__ == '__main__':
	#configurar latitude: export PX4_HOME_LAT=-26.236411
	#configurar longitude: export PX4_HOME_LON=-42.963770
	#lancar simulacao (h480): make px4_sitl gazebo_typhoon_h480
	#lancar simulacao (iris): make px4_sitl gazebo
	#iniciar MavROS com SITL: roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
	#topicos: rostopics list
	
	lon_final=-42.9640769
	lat_final=-26.2382942

	d=30
	h=30
	
	gps=GPS()
	(lat_inicial, lon_inicial)=gps.le_gps()
	(L,tetag,lon1,lat1,lon2,lat2)=calcula_pontos(lat_inicial, lon_inicial, lat_final, lon_final, d)
	
	wps=[(lon_inicial,lat_inicial),(lon1,lat1),(lon2,lat2),(lon_final,lat_final)]

	rospy.init_node('waypoint_node', anonymous=True)
	pub = rospy.Publisher('global',String,queue_size=10)
	autor = "Gabryel"
	pub.publish(autor)	
	cria_waypoints(wps,h)

	#print(lat_inicial)
	#print(lon_inicial)
	#print(lat_final)
	#print(lon_final)
	#print(L)
	#print(tetag)
	#print(lat1)
	#print(lon1)
	#print(lat2)
	#print(lon2)

