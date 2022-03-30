import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget, MountControl
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String
import time
from pyquaternion import Quaternion
import math
import threading
import numpy as np

import sys
#necessario para que o sistema não tente importar o opencv do ROS
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


class Px4Controller:

    def __init__(self):

        self.imu = None
        self.gps = None
        self.local_pose = None
        self.current_state = None
        self.current_heading = None
        self.takeoff_height = 10.0
        self.local_enu_position = None

        self.cur_target_pose = None
        self.global_target = None
        
        self.velocidade = None

        self.received_new_task = False
        self.arm_state = False
        self.offboard_state = False
        self.received_imu = False
        self.frame = "BODY"

        self.state = None
        
        self.controle_gimbal = None

        '''
        subscribers do ROS
        '''
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        self.mavros_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback)
        self.gps_sub = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)

        self.set_target_position_sub = rospy.Subscriber("gi/set_pose/position", PoseStamped, self.set_target_position_callback)
        self.set_target_yaw_sub = rospy.Subscriber("gi/set_pose/orientation", Float32, self.set_target_yaw_callback)
        self.custom_activity_sub = rospy.Subscriber("gi/set_activity/type", String, self.custom_activity_callback)
        #self.leitura_camera = rospy.Subscriber('camera_opf', String, self.cls_camera_callback)


        '''
        publishers do ROS
        '''
        self.local_target_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.setpoint_velocity_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size = 10)
        #self.controle_gimbal_pub = rospy.Publisher('mavros/mount_control/command', MountControl, queue_size = 10)
        
        '''
        servicos do ROS
        '''
        self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)


        print("Controlador Px4 Inicializado!")


    def start(self):
        #rospy.init_node("offboard_node")
        for i in range(10):
            if self.current_heading is not None:
                break
            else:
                print("Aguardando Inicializacao.")
                time.sleep(0.5)
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

        #print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

        for i in range(10):
            self.local_target_pub.publish(self.cur_target_pose)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()
            time.sleep(0.2)


        if self.takeoff_detection():
            print("Veiculo Decolou!")

        else:
            print("Falha ao Decolar!")
            return

        '''
        Thread principal do ROS
        '''
        while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):

            self.local_target_pub.publish(self.cur_target_pose)

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                if(self.disarm()):

                    self.state = "DISARMED"


            time.sleep(0.1)


    def construct_target(self, x, y, z, yaw, yaw_rate = 0.05):
        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = 1

        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z
        
        # target_raw_pose.velocity.x = vx
        # target_raw_pose.velocity.y = vy
        # target_raw_pose.velocity.z = vz
        
        # target_raw_pose.acceleration_or_force.x = afx
        # target_raw_pose.acceleration_or_force.y = afy
        # target_raw_pose.acceleration_or_force.z = afz

        target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        target_raw_pose.yaw = yaw
        target_raw_pose.yaw_rate = yaw_rate

        return target_raw_pose
    
    def controle_velocidade(self, v, yaw, yaw_rate=0.05):
        drone_vel = PositionTarget()
        
        vx = v*np.cos(yaw)
        vy = v*np.sin(yaw)
        drone_vel.velocity.x = vx
        drone_vel.velocity.y = vy

        drone_vel.header.stamp = rospy.Time.now()

        drone_vel.coordinate_frame = 1
        
        #z=30
        #drone_vel.position.z = z
        
        drone_vel.velocity.x = vx
        drone_vel.velocity.y = vy
        # target_raw_pose.velocity.z = vz
        
        # target_raw_pose.acceleration_or_force.x = afx
        # target_raw_pose.acceleration_or_force.y = afy
        # target_raw_pose.acceleration_or_force.z = afz

        drone_vel.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        drone_vel.yaw = yaw
        drone_vel.yaw_rate = yaw_rate
        
        return drone_vel
    
    def controle_gimbal_camera(self, r, p, y):
        gb_controle = MountControl()
        
        gb_controle.header.stamp = rospy.Time.now()
        gb_controle.header.frame_id="map"
        gb_controle.mode=2
        
        gb_controle.roll=r
        gb_controle.pitch=p
        gb_controle.yaw=y
        
        return gb_controle
        



    '''
    cur_p : poseStamped
    target_p: positionTarget
    '''
    def position_distance(self, cur_p, target_p, threshold=0.1):
        delta_x = math.fabs(cur_p.pose.position.x - target_p.position.x)
        delta_y = math.fabs(cur_p.pose.position.y - target_p.position.y)
        delta_z = math.fabs(cur_p.pose.position.z - target_p.position.z)

        if (delta_x + delta_y + delta_z < threshold):
            return True
        else:
            return False


    def local_pose_callback(self, msg):
        self.local_pose = msg
        self.local_enu_position = msg


    def mavros_state_callback(self, msg):
        self.mavros_state = msg.mode


    def imu_callback(self, msg):
        global global_imu, current_heading
        self.imu = msg

        self.current_heading = self.q2yaw(self.imu.orientation)

        self.received_imu = True


    def gps_callback(self, msg):
        self.gps = msg

    def FLU2ENU(self, msg):

        FLU_x = msg.pose.position.x * math.cos(self.current_heading) - msg.pose.position.y * math.sin(self.current_heading)
        FLU_y = msg.pose.position.x * math.sin(self.current_heading) + msg.pose.position.y * math.cos(self.current_heading)
        FLU_z = msg.pose.position.z

        return FLU_x, FLU_y, FLU_z


    def set_target_position_callback(self, msg):
        print("Nova Tarefa de Posicionamento Recebida!")

        if msg.header.frame_id == 'base_link':
            '''
            BODY_FLU
            '''
            # For Body frame, we will use FLU (Forward, Left and Up)
            #           +Z     +X
            #            ^    ^
            #            |  /
            #            |/
            #  +Y <------body

            self.frame = "BODY"

            print("Frame FLU")

            ENU_X, ENU_Y, ENU_Z = self.FLU2ENU(msg)

            ENU_X = ENU_X + self.local_pose.pose.position.x
            ENU_Y = ENU_Y + self.local_pose.pose.position.y
            ENU_Z = ENU_Z + self.local_pose.pose.position.z

            self.cur_target_pose = self.construct_target(ENU_X,
                                                         ENU_Y,
                                                         ENU_Z,
                                                         self.current_heading)


        else:
            '''
            LOCAL_ENU
            '''
            # For world frame, we will use ENU (EAST, NORTH and UP)
            #     +Z     +Y
            #      ^    ^
            #      |  /
            #      |/
            #    world------> +X

            self.frame = "LOCAL_ENU"
            print("Frame local ENU")

            self.cur_target_pose = self.construct_target(msg.pose.position.x,
                                                         msg.pose.position.y,
                                                         msg.pose.position.z,
                                                         self.current_heading)

    '''
     Recebeu Atividade Custom
     '''

    def custom_activity_callback(self, msg):

        print("Received Custom Activity:", msg.data)

        if msg.data == "LAND":
            print("ATERRISANDO!")
            self.state = "LAND"
            self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                         self.local_pose.pose.position.y,
                                                         0.1,
                                                         self.current_heading)

        if msg.data == "HOVER":
            print("HOVERING!")
            self.state = "HOVER"
            self.hover()

        else:
            print("Received Custom Activity:", msg.data, "not supported yet!")


    def set_target_yaw_callback(self, msg):
        print("Received New Yaw Task!")

        yaw_deg = msg.data * math.pi / 180.0
        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     yaw_deg)

    '''
    return yaw from current IMU
    '''
    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad


    def arm(self):
        if self.armService(True):
            return True
        else:
            print("Vehicle arming failed!")
            return False

    def disarm(self):
        if self.armService(False):
            return True
        else:
            print("Vehicle disarming failed!")
            return False


    def offboard(self):
        if self.flightModeService(custom_mode='OFFBOARD'):
            return True
        else:
            print("Vechile Offboard failed")
            return False


    def hover(self):

        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     self.current_heading)

    def takeoff_detection(self):
        if self.local_pose.pose.position.z > 0.2 and self.offboard_state and self.arm_state:
            return True
        else:
            return False

    def navegue(self, pontos,j=0):
	#rospy.init_node("offboard_node")
        for i in range(10):
            if self.current_heading is not None:
                break
            else:
                print("Waiting for initialization.")
                time.sleep(0.5)
        heading_inicio=self.current_heading
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height*10, self.current_heading)
	
        #print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

        for i in range(10):
            self.local_target_pub.publish(self.cur_target_pose)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()
            time.sleep(0.2)


        if self.takeoff_detection():
            print("Vehicle Took Off!")

        else:
            print("Vehicle Took Off Failed!")
            return

        '''
        main ROS thread
        '''
        while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
            video = cv2.VideoCapture("udp://127.0.0.1:14557?overrun_nonfatal=1&fifo_size=50000000")
            cv2.imshow(video)
            self.local_target_pub.publish(self.cur_target_pose)
            if j < len(pontos):
            			xt=pontos[j][0]
            			yt=pontos[j][1]
            			zt=pontos[j][2]
            			yawt=np.float32(pontos[j][3]*np.pi/180)
            			self.cur_target_pose = self.construct_target(xt,yt,zt,yawt)
            			self.local_target_pub.publish(self.cur_target_pose)
            			x=self.local_pose.pose.position.x
            			y=self.local_pose.pose.position.y
            			z=self.local_pose.pose.position.z
            			yaw=self.current_heading
            if np.absolute(x-xt)<=2 and np.absolute(y-yt)<=2 and np.absolute(z-zt)<=2:
             			j=j+1
             			print(j)
            		#else:
            			#self.hover()
            		#time.sleep(0.1)	
            		#rospy.spin()

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                	if(self.disarm()):

                    		self.state = "DISARMED"


            time.sleep(0.1)
            print(self.current_heading)
    
    
    def navegue2(self, v, yaw, j=0):
        
        #rospy.init_node("offboard_node")
        for i in range(10):
            if self.current_heading is not None:
                print("passou aqui")
                break
            else:
                print("Aguardando Inicializacao")
                time.sleep(0.5)
                
        heading_inicio=self.current_heading
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
	
        #print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

        #for i in range(20):
        while self.local_pose.pose.position.z < self.takeoff_height:
            self.local_target_pub.publish(self.cur_target_pose)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()
            #time.sleep(0.2)
            print("to aqui")


        if self.takeoff_detection():
            print("Veiculo Decolou!")

        else:
            print("Falha ao Decolar!")
            return

        '''
        Principal Thread do ROS
        '''
        while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
            self.local_target_pub.publish(self.cur_target_pose)
            self.velocidade = self.controle_velocidade(v, np.radians(yaw))
            self.setpoint_velocity_pub.publish(self.velocidade)
            time.sleep(0.2)
            		#else:
            			#self.hover()
            		#time.sleep(0.1)	
            		#rospy.spin()

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                	if(self.disarm()):

                    		self.state = "DISARMED"

            #self.controle_gimbal=self.controle_gimbal_camera(0, -90, 0)
            #self.controle_gimbal_pub.publish(self.controle_gimbal)
        
            time.sleep(0.1)
            print(self.current_heading*180/np.pi)
msg=None

def callback(data):
	global msg
	msg=data.data

rospy.init_node("offboard_node")

if __name__ == '__main__':
    h=10
    con = Px4Controller()
    v=2
    ang=-90
    con.navegue2(v, ang)

