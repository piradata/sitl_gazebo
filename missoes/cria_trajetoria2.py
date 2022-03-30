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
import matplotlib
import matplotlib.pyplot as plt

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
		altitude = self.gps.altitude
		return(latitude,longitude, altitude)

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

def direcao_destino(lir, loir, lfr, lofr):

	if lir>lfr:
		dy="S"
	elif lir<lfr:
		dy="N"
	else:
		dy="M"

	if loir>lofr:
		dx="O"
	elif loir<lofr:
		dx="L"
	else:
		dx="M"

	direc=dy+dx
	return direc


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

def trajetoria(lir,loir,lfr,lofr):
	d=30
	direc=direcao_destino(lir,loir,lfr,lofr)
	(L, tetag, lon1, lat1, lon2, lat2)=calcula_pontos(lir,loir,lfr,lofr,d)
	teta=tetag*np.pi/180
	if direc=="ML":
		C=[L/2,0];
		x=np.arange(0, L, L/100).tolist()
		y=np.zeros(len(x))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(x)):
			y[i]=np.sqrt((L/2)**2-(x[i]-C[0])**2)+C[1]
			if i==0:
				yaw[i]=0
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(x,L)
		y=y.tolist()
		yf=0
		np.append(y,0)
		y[len(y)-1]=yf
		yaw=yaw.tolist()
		yawf=-90
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	elif direc=="MO":
		C=[-L/2,0];
		x=np.arange(0, -L, -L/100).tolist()
		y=np.zeros(len(x))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(x)):
			y[i]=np.sqrt((L/2)**2-(x[i]-C[0])**2)+C[1]
			if i==0:
				yaw[i]=-180
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(x,-L)
		y=y.tolist()
		yf=0
		np.append(y,0)
		y[len(y)-1]=yf
		yaw=yaw.tolist()
		yawf=-270
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	elif direc=="NM":
		C=[0, L/2];
		y=np.arange(0, L, L/100).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[1])**2)+C[0]
			if i==0:
				yaw[i]=90
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,L)
		x=x.tolist()
		xf=0
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=0
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	elif direc=="SM":
		C=[0, -L/2];
		y=np.arange(0, -L, -L/100).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[1])**2)+C[0]
			if i==0:
				yaw[i]=270
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,-L)
		x=x.tolist()
		xf=0
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=180
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	elif direc=="SO":
		C=[-L/2*np.cos(teta), -L/2*np.sin(teta)];
		y=np.arange(0, -L*np.sin(teta), (-L*np.sin(teta)/100)).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[1])**2)+C[0]
			if i==0:
				yaw[i]=180+tetag
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,-L*np.sin(teta))
		x=x.tolist()
		xf=(-L*np.cos(teta))
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=90+tetag
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf
		
	elif direc=="NL":
		C=[L/2*np.sin(teta), L/2*np.cos(teta)];
		y=np.arange(0, L*np.cos(teta), (L*np.cos(teta)/100)).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[1])**2)+C[0]
			if i==0:
				yaw[i]=90-tetag
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,L*np.cos(teta))
		x=x.tolist()
		xf=(L*np.sin(teta))
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=-tetag
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf
		

	elif direc=="SL":
		C=[L/2*np.cos(teta), -L/2*np.sin(teta)];
		y=np.arange(0, -L*np.sin(teta), (-L*np.sin(teta)/100)).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[0])**2)+C[1]
			if i==0:
				yaw[i]=-tetag
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,-L*np.sin(teta))
		x=x.tolist()
		xf=(L*np.cos(teta))
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=-tetag-90
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	elif direc=="NO":
		C=[-L/2*np.sin(teta), L/2*np.cos(teta)];
		y=np.arange(0, L*np.cos(teta), (L*np.cos(teta)/100)).tolist()
		x=np.zeros(len(y))
		yaw=np.zeros(len(x))
		Dyaw=np.float32(90.0/len(x))
		for i in range(len(y)):
			x[i]=np.sqrt((L/2)**2-(y[i]-C[0])**2)+C[1]
			if i==0:
				yaw[i]=90+tetag
			else:
				yaw[i]=yaw[i-1]-Dyaw
		np.append(y,L*np.cos(teta))
		x=x.tolist()
		xf=(-L*np.sin(teta))
		np.append(x,0)
		x[len(x)-1]=xf
		yaw=yaw.tolist()
		yawf=tetag
		np.append(yaw,0)
		yaw[len(yaw)-1]=yawf

	plt.plot(x,y)
	plt.savefig('rota.png')
	return(x,y,yaw,C)

def trajetoria2(lir,loir,lfr,lofr, des, pos, lnavio):
    d=30
    direc=direcao_destino(lir,loir,lfr,lofr)
    (L, tetag, lon1, lat1, lon2, lat2)=calcula_pontos(lir,loir,lfr,lofr,d)
    teta=tetag*np.pi/180
    
    if pos=="proa":
        if des=="meianau":
            L=L+lnavio/2
        elif des=="popa":
            L=L+lnavio
    elif pos=="meianau":
        if des=="proa":
            L=L-lnavio/2
        elif des=="popa":
            L=L+lnavio/2
    elif pos=="popa":
        if des=="proa":
            L=L-lnavio
        elif des=="meianau":
            L=L-lnavio/2
    elif pos==des:
        pass
    
    if direc=="ML":  
        C=[L,0]
        x=np.arange(0, L, L/100).tolist()
        y=np.zeros(len(x))
        yaw=np.zeros(len(x))
        Dyaw=np.float32(90.0/len(x))
        for i in range(len(x)):
            y[i]=np.sqrt((L)**2-(x[i]-C[0])**2)+C[1]
            if i==0:
                yaw[i]=0
            else:
                yaw[i]=yaw[i-1]-Dyaw
        np.append(x,L)
        y=y.tolist()
        yf=L
        np.append(y,0)
        y[len(y)-1]=yf
        yaw=yaw.tolist()
        yawf=-90
        np.append(yaw,0)
        yaw[len(yaw)-1]=yawf
   
    elif direc=="MO":
        C=[-L,0]
        x=np.arange(0, -L, -L/100).tolist()
        y=np.zeros(len(x))
        yaw=np.zeros(len(x))
        Dyaw=np.float32(90.0/len(x))
        for i in range(len(x)):
            y[i]=np.sqrt((L)**2-(x[i]-C[0])**2)+C[1]
            if i==0:
                yaw[i]=-180
            else:
                yaw[i]=yaw[i-1]-Dyaw
        np.append(x,-L)
        y=y.tolist()
        yf=-L
        np.append(y,0)
        y[len(y)-1]=yf
        yaw=yaw.tolist()
        yawf=-270
        np.append(yaw,0)
        yaw[len(yaw)-1]=yawf
   
    elif direc=="NM":
   		C=[0, L]
   		y=np.arange(0, L, L/100).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[1])**2)+C[0]
   			if i==0:
   				yaw[i]=90
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,L)
   		x=x.tolist()
   		xf=-L
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=0
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   
    elif direc=="SM":
   		C=[0, -L]
   		y=np.arange(0, -L, -L/100).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[1])**2)+C[0]
   			if i==0:
   				yaw[i]=270
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,-L)
   		x=x.tolist()
   		xf=L
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=180
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   
    elif direc=="SO":
   		C=[-L*np.cos(teta), -L*np.sin(teta)]
   		y=np.arange(0, -L*np.sin(teta), (-L*np.sin(teta)/100)).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[1])**2)+C[0]
   			if i==0:
   				yaw[i]=180+tetag
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,y[len(y)-1])
   		x=x.tolist()
   		xf=x[len(x)-1]
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=90+tetag
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   		
    elif direc=="NL":
   		C=[L*np.sin(teta), L*np.cos(teta)]
   		y=np.arange(0, L*np.cos(teta), (L*np.cos(teta)/100)).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[1])**2)+C[0]
   			if i==0:
   				yaw[i]=90-tetag
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,y[len(y)-1])
   		x=x.tolist()
   		xf=x[len(x)-1]
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=-tetag
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   		
   
    elif direc=="SL":
   		C=[L*np.cos(teta), -L*np.sin(teta)]
   		y=np.arange(0, -L*np.sin(teta), (-L*np.sin(teta)/100)).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[0])**2)+C[1]
   			if i==0:
   				yaw[i]=-tetag
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,y[len(y)-1])
   		x=x.tolist()
   		xf=x[len(x)-1]
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=-tetag-90
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   
    elif direc=="NO":
   		C=[-L*np.sin(teta), L*np.cos(teta)]
   		y=np.arange(0, L*np.cos(teta), (L*np.cos(teta)/100)).tolist()
   		x=np.zeros(len(y))
   		yaw=np.zeros(len(x))
   		Dyaw=np.float32(90.0/len(x))
   		for i in range(len(y)):
   			x[i]=np.sqrt((L)**2-(y[i]-C[0])**2)+C[1]
   			if i==0:
   				yaw[i]=90+tetag
   			else:
   				yaw[i]=yaw[i-1]-Dyaw
   		np.append(y,y[len(y)-1])
   		x=x.tolist()
   		xf=x[len(x)-1]
   		np.append(x,0)
   		x[len(x)-1]=xf
   		yaw=yaw.tolist()
   		yawf=tetag
   		np.append(yaw,0)
   		yaw[len(yaw)-1]=yawf
   
    plt.plot(x,y)
    plt.savefig('rota2.png')
    return(x,y,yaw,C)

def configura_missao(waypoints):
     pub = rospy.Publisher('configura_missao', String, queue_size=10)
     rate = rospy.Rate(10) # 10hz
     while not rospy.is_shutdown():
         configuracao = str(waypoints).strip('[]')
         rospy.loginfo(configuracao)
         pub.publish(configuracao)
         rate.sleep()

def configura_trajetoria(pontos):
     pub = rospy.Publisher('configura_trajetoria', String, queue_size=10)
     rate = rospy.Rate(10) # 10hz
     while not rospy.is_shutdown():
         configuracao = str(pontos).strip('[]')
         rospy.loginfo(configuracao)
         pub.publish(configuracao)
         rate.sleep()

if __name__ == '__main__':
	#configurar latitude: export PX4_HOME_LAT=-26.236411
	#configurar longitude: export PX4_HOME_LON=-42.963770
	#lancar simulacao (h480): make px4_sitl gazebo_typhoon_h480
	#lancar simulacao (iris): make px4_sitl gazebo
	#iniciar MavROS com SITL: roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
	#topicos: rostopics list
	
    #destino
    des="meianau"
    #posição da coordenada de destino
	#proa
    pos="proa"
    lon_final=-42.9640769
    lat_final=-26.2382942
    
    #comprimento do navio
    lnavio=310

    d=30
    h=30
	
    gps=GPS()
    (lat_inicial, lon_inicial, alt_inicial)=gps.le_gps()
    #(L,tetag,lon1,lat1,lon2,lat2)=calcula_pontos(lat_inicial, lon_inicial, lat_final, lon_final, d)
	
	#wps=[(lon_inicial,lat_inicial),(lon1,lat1),(lon2,lat2),(lon_final,lat_final), alt_inicial]
    (x,y,yaw,C)=trajetoria2(lat_inicial, lon_inicial, lat_final, lon_final, des, pos, lnavio)
    direc=direcao_destino(lat_inicial, lon_inicial, lat_final, lon_final)
    pontos=[x,y,yaw]

    try:
	    #configura_missao(wps)
	    #(x,y,C)=trajetoria(lat_inicial, lon_inicial, lat_final, lon_final)
	    configura_trajetoria(pontos)
    except rospy.ROSInterruptException:
	    pass



