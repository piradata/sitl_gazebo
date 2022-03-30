import rospy
import mavros
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import Waypoint, WaypointList 
from mavros_msgs.srv import  WaypointPull, WaypointPush, WaypointClear, WaypointSetCurrent

def cria_waypoints(dados, altitude, altura):
    limpa_waypoints()
    wl = WaypointList()
    wp = Waypoint()
    fr=3
    wp.frame = fr
    wp.command = 22  # decola
    wp.is_current = True
    wp.autocontinue = True
    wp.param1 = altitude  # altitude de decolagem
    wp.param2 = 0
    wp.param3 = 0
    wp.param4 = 0
    wp.x_lat = dados[1]
    wp.y_long = dados[0]
    wp.z_alt = altitude
    wl.waypoints.append(wp)
    for i in range(4):
        wp = Waypoint()
        wp.frame = fr
        wp.command = 16  # navega para o ponto
        wp.is_current = False
        wp.autocontinue = True
        wp.param1 = 0 
        wp.param2 = 0
        wp.param3 = 0
        wp.param4 = 0
	wp.x_lat = dados[2*i+1]
	wp.y_long = dados[2*i]
	wp.z_alt = (altitude+altura)
	wl.waypoints.append(wp)
    try:
        service = rospy.ServiceProxy('mavros/mission/push', WaypointPush, persistent=False)
	service(start_index=np.uint32(0), waypoints=wl.waypoints)
        if service.call(wl.waypoints).success:
            print 'missao escrita'
        else:
            print 'erro ao escrever missao'
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def limpa_waypoints():
        try:
            response = rospy.ServiceProxy(
                'mavros/mission/clear', WaypointClear)
            return response.call().success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False

msg=None

def callback(data):
	global msg
	msg=data.data
	#rospy.loginfo("%s", msg)

rospy.init_node('waypoint_node', anonymous=True)

def cria_arquivo_missao(dados, altitude, altura):
	fr=3
	parametros=(altura,fr,dados[1],dados[0],altura,altura,fr,dados[3],dados[2],altura,altura,fr,dados[5],dados[4],altura,altura,fr,dados[7],dados[6],altura,dados[1],dados[0],altitude)
	arquivo=('{\n"fileType": "Plan",\n"geoFence": {\n"circles": [\n],\n"polygons": [\n],\n"version": 2\n},\n"groundStation":\n"QGroundControl",\n"mission": {\n"cruiseSpeed": 15,\n"firmwareType": 12,\n"hoverSpeed": 5,\n"items": [\n{\n"AMSLAltAboveTerrain": null,\n"Altitude": %d,\n"AltitudeMode": 1,\n"autoContinue": true,\n"command": 22,\n"doJumpId": 1,\n"frame": %d,\n"params": [\n15,\n0,\n0,\nnull,\n%f,\n%f,\n%d\n],\n"type": "SimpleItem"\n},\n{\n"AMSLAltAboveTerrain": null,\n"Altitude": %d,\n"AltitudeMode": 1,\n"autoContinue": true,\n"command": 16,\n"doJumpId": 2,\n"frame": %d,\n"params": [\n0,\n0,\n0,\nnull,\n%f,\n%f,\n%d\n],\n"type": "SimpleItem"\n},\n{\n"AMSLAltAboveTerrain": null,\n"Altitude": %d,\n"AltitudeMode": 1,\n"autoContinue": true,\n"command": 16,\n"doJumpId": 3,\n"frame": %d,\n"params": [\n0,\n0,\n0,\nnull,\n%f,\n%f,\n%d\n],\n"type": "SimpleItem"\n},\n{\n"AMSLAltAboveTerrain": null,\n"Altitude": %d,\n"AltitudeMode": 1,\n"autoContinue": true,\n"command": 16,\n"doJumpId": 3,\n"frame": %d,\n"params": [\n0,\n0,\n0,\nnull,\n%f,\n%f,\n%d\n],\n"type": "SimpleItem"\n}\n],\n"plannedHomePosition": [\n%f,\n%f,\n%f\n],\n"vehicleType": 2,\n"version": 2\n},\n"rallyPoints": {\n"points": [\n],\n"version": 2\n},\n"version": 1\n}' % parametros)
	print(arquivo)
	f= open("missao.plan","w+")
	f.write(arquivo)
	f.close

if __name__ == '__main__':
	while msg is None:
		rospy.Subscriber("configura_missao", String, callback)
		if msg is not None:
			dados=msg	
	dados=dados.replace(")","")
	dados=dados.replace("(","")
	dados = [float(item) for item in dados.split(",")]
	alt_inicial=dados[-1]	
	print(alt_inicial)
	h=30
	cria_arquivo_missao(dados, alt_inicial, h)
	cria_waypoints(dados, alt_inicial, h)

