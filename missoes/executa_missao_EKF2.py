import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String
import time
from pyquaternion import Quaternion
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from tinyekf import EKF

class DroneEKF(EKF):
    '''
    EKF para auxiliar navegação do drone
    '''

    def __init__(self):

        # 6 estados (x, y, yaw, omega, v, tempo), 6 medições, alto erro
        EKF.__init__(self, 1, 2, rval=.8, qval=.8, pval=.8)

    def f(self, Tk):

        return np.copy(Tk), np.eye(1)


    def h(self, Tk):
        h = np.array([Tk[0], Tk[0]])
        H = np.array([[1], [1]])

        return h, H


class Px4Controller:

    def __init__(self):

        self.imu = None
        self.gps = None
        self.local_pose = None
        self.current_state = None
        self.current_heading = None
        self.takeoff_height = 2.5
        self.local_enu_position = None

        self.cur_target_pose = None
        self.global_target = None

        self.camera_msg = None
        self.destino_msg = None
        self.velocidade_drone = None
        self.leitura_velocidade_drone=None

        self.received_new_task = False
        self.arm_state = False
        self.offboard_state = False
        self.received_imu = False
        self.frame = "BODY"

        self.state = None

        '''
        ros subscribers
        '''
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        self.mavros_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback)
        self.gps_sub = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)

        self.set_target_position_sub = rospy.Subscriber("gi/set_pose/position", PoseStamped, self.set_target_position_callback)
        self.set_target_yaw_sub = rospy.Subscriber("gi/set_pose/orientation", Float32, self.set_target_yaw_callback)
        self.custom_activity_sub = rospy.Subscriber("gi/set_activity/type", String, self.custom_activity_callback)

        self.leitura_camera = rospy.Subscriber('dados_camera_kf', String, self.cls_camera_callback)
        self.config_destino = rospy.Subscriber("configura_centro", String, self.destino_callback)
        self.velocidade_sub = rospy.Subscriber('/mavros/local_position/velocity', TwistStamped, self.velocidade_callback)

        '''
        ros publishers
        '''
        self.local_target_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.setpoint_velocidade_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size = 10)

        '''
        ros services
        '''
        self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)


        print("Px4 Controller Initialized!")


    def start(self):
        #rospy.init_node("offboard_node")
        for i in range(10):
            if self.current_heading is not None:
                break
            else:
                print("Waiting for initialization.")
                time.sleep(0.5)
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

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

            self.local_target_pub.publish(self.cur_target_pose)

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                if(self.disarm()):

                    self.state = "DISARMED"


            time.sleep(0.1)

# std_msgs/Header header

# uint8 coordinate_frame
# uint8 FRAME_LOCAL_NED = 1
# uint8 FRAME_LOCAL_OFFSET_NED = 7
# uint8 FRAME_BODY_NED = 8
# uint8 FRAME_BODY_OFFSET_NED = 9

# uint16 type_mask
# uint16 IGNORE_PX = 1 # Position ignore flags
# uint16 IGNORE_PY = 2
# uint16 IGNORE_PZ = 4
# uint16 IGNORE_VX = 8 # Velocity vector ignore flags
# uint16 IGNORE_VY = 16
# uint16 IGNORE_VZ = 32
# uint16 IGNORE_AFX = 64 # Acceleration/Force vector ignore flags
# uint16 IGNORE_AFY = 128
# uint16 IGNORE_AFZ = 256
# uint16 FORCE = 512 # Force in af vector flag
# uint16 IGNORE_YAW = 1024
# uint16 IGNORE_YAW_RATE = 2048

# geometry_msgs/Point position
# geometry_msgs/Vector3 velocity
# geometry_msgs/Vector3 acceleration_or_force
# float32 yaw
# float32 yaw_rate


    def construct_target(self, x, y, z, yaw, yaw_rate = 0.00):
        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = 9

        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z

        # target_raw_pose.velocity.x = 1
        # target_raw_pose.velocity.y = 1
        # target_raw_pose.velocity.z = 1

        # target_raw_pose.acceleration_or_force.x = 0.01
        # target_raw_pose.acceleration_or_force.y = 0.01
        # target_raw_pose.acceleration_or_force.z = 0.01

        target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        target_raw_pose.yaw = yaw
        target_raw_pose.yaw_rate = yaw_rate

        return target_raw_pose

    def controle_orientacao(self, yaw, h, yaw_rate = 0.00):
        print('ENTROU NO CONTROLE DE ORIENTACAO')
        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = 9

        target_raw_pose.position.x = self.local_pose.pose.position.x
        target_raw_pose.position.y = self.local_pose.pose.position.y
        target_raw_pose.position.z = h


        target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        target_raw_pose.yaw = np.radians(yaw)
        target_raw_pose.yaw_rate = yaw_rate

        return target_raw_pose

    def calcula_velocidade_drone(self, v, ang):
        #ang=self.current_heading*180/np.pi
        print('ENTROU NO CALCULO DE VELOCIDADE')
        if ang==0:
            orientacao='L'
            vx=v*np.cos(np.radians(ang))
            vy=v*np.sin(np.radians(ang))
        elif ang>0 and ang<90:
            orientacao='NE'
            vx=v*np.cos(np.radians(ang))
            vy=v*np.sin(np.radians(ang))
        elif ang==90:
            orientacao='N'
            vx=v*np.cos(np.radians(ang))
            vy=v*np.sin(np.radians(ang))
        elif ang>90 and ang<180:
            orientacao='NO'
            ang=180-ang
            vx=-1*v*np.cos(np.radians(ang))
            vy=v*np.sin(np.radians(ang))
        elif ang==180 or ang==-180:
            orientacao='O'
            vx=-1*v*np.cos(np.radians(ang))
            vy=v*np.sin(np.radians(ang))
        elif ang<0 and ang>-90:
            orientacao='SE'
            ang=-1*ang
            vx=v*np.cos(np.radians(ang))
            vy=-1*v*np.sin(np.radians(ang))
        elif ang==-90:
            orientacao='S'
            ang=-1*ang
            vx=-1*v*np.cos(np.radians(ang))
            vy=-1*v*np.sin(np.radians(ang))
        else:
            orientacao='SO'
            ang=-1*(-180-ang)
            vx=-1*v*np.cos(np.radians(ang))
            vy=-1*v*np.sin(np.radians(ang))
        return orientacao, vx, vy

    def controle_velocidade(self, vx, vy, yaw_rate=0.00):
            print('ENTROU NO CONTROLE DE VELOCIDADE')
            drone_vel = PositionTarget()

            #yaw=yaw*np.pi/180

            #vx = v*np.cos(yaw)
            #vy = v*np.sin(yaw)
            #drone_vel.velocity.x = vx
            #drone_vel.velocity.y = vy

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
                                        + PositionTarget.IGNORE_YAW + PositionTarget.IGNORE_YAW_RATE + PositionTarget.FORCE

            #drone_vel.yaw = self.current_heading
            #drone_vel.yaw_rate = yaw_rate
            #print(vx)
            #print(vy)

            return drone_vel

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

    def cls_camera_callback(self, data):
        self.camera_msg = data.data
        #camera_msg=data.data
        if self.camera_msg=='Placa Encontrada!':
            self.camera_msg='Parando Etapa 2!'
        else:
            self.camera_msg=self.camera_msg.replace("[","")
            self.camera_msg=self.camera_msg.replace("]","")
            self.camera_msg = [float(item_camera) for item_camera in self.camera_msg.split(",")]

    def destino_callback(self, data):
        self.destino_msg=data.data
        self.destino_msg = [float(coordenadas_centro) for coordenadas_centro in self.destino_msg.split(",")]

    def local_pose_callback(self, msg):
        self.local_pose = msg
        self.local_enu_position = msg

    def velocidade_callback(self, msg):
        self.leitura_velocidade_drone = msg


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
        print("Received New Position Task!")

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

            print("body FLU frame")

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
            print("local ENU frame")

            self.cur_target_pose = self.construct_target(msg.pose.position.x,
                                                         msg.pose.position.y,
                                                         msg.pose.position.z,
                                                         self.current_heading)

    '''
     Receive A Custom Activity
     '''

    def custom_activity_callback(self, msg):

        print("Received Custom Activity:", msg.data)

        if msg.data == "LAND":
            print("LANDING!")
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


    def hover(self, h):

        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     self.current_heading)
    def fixa(self, status_decolagem, h):
        if status_decolagem==True:
            #rospy.init_node("offboard_node")
            for i in range(10):
                if self.current_heading is not None:
                    break
                else:
                    print("Waiting for initialization.")
                    time.sleep(0.5)
                heading_inicio=self.current_heading
                self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

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
            arma e decola a primeira vez
            '''
            if self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
                self.local_target_pub.publish(self.cur_target_pose)
                self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                         self.local_pose.pose.position.y,
                                                         h,
                                                         self.current_heading)
                self.local_target_pub.publish(self.cur_target_pose)
                dados_camera=self.camera_msg
                print(dados_camera)
                desx=self.destino_msg[0]
                desy=self.destino_msg[1]
                print(desx, desy)

        else:
            '''
            se mantem na posição
            '''
            if self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
                self.local_target_pub.publish(self.cur_target_pose)
                self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                         self.local_pose.pose.position.y,
                                                         h,
                                                         self.current_heading)
                self.local_target_pub.publish(self.cur_target_pose)
                dados_camera=self.camera_msg
                print(dados_camera)
                desx=self.destino_msg[0]
                desy=self.destino_msg[1]
                print(desx, desy)

    def takeoff_detection(self):
        if self.local_pose.pose.position.z > 0.1 and self.offboard_state and self.arm_state:
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
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

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
            print(self.current_heading*180/np.pi)

    def missao_pt1(self, pontos,j=0):
	#rospy.init_node("offboard_node")
        for i in range(10):
            if self.current_heading is not None:
                break
            else:
                print("Waiting for initialization.")
                time.sleep(0.5)
        heading_inicio=self.current_heading
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

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
        log_altura=[]
        log_yaw=[]
        log_tempo=[]
        log_x=[]
        log_y=[]
        while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
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
                t=rospy.get_time()
                log_altura.append(z)
                log_yaw.append(yaw*180/np.pi)
                log_tempo.append(t)
                log_x.append(x)
                log_y.append(y)
            if np.absolute(x-xt)<=2 and np.absolute(y-yt)<=2 and np.absolute(z-zt)<=2:
             			j=j+1
             			print(j)

            if j > len(pontos):
                log_etapa1={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
                log_etapa1 = pd.DataFrame(data=log_etapa1)
                #log_etapa1.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa1.csv', index=False)
                log_etapa1.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa1.csv', index=False)

                plt.plot(log_tempo,log_altura)
                plt.xlabel('Tempo (s)')
                plt.ylabel('Altura (m)')
                plt.title('Altura x Tempo de Simulacao')
                plt.grid()
                #plt.savefig('log_AlturaxTempo.png')

                plt.clf()
                plt.plot(log_tempo,log_yaw)
                plt.xlabel('Tempo (s)')
                plt.ylabel('Orientacao (º)')
                plt.title('Orientacao x Tempo de Simulacao')
                plt.grid()
                #plt.savefig('log_YawxTempo.png')

                plt.clf()
                fig=plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot3D(log_x,log_y,log_altura)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Altura (m)')
                ax.set_title('Orientacao x Tempo de Simulacao')
                #fig.savefig('log_posicaoDrone_etapa1.png')

                break

            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                	if(self.disarm()):

                    		self.state = "DISARMED"
            time.sleep(0.1)
            print(self.current_heading*180/np.pi)

    def missao_pt2(self, status_decolagem, h=15):
        vel=1.0
        max_dist=5
        #centro=img_x/2
        #tolerancia=0.1*img_x
        kp=1
        if status_decolagem==True:
            #rospy.init_node("offboard_node")
            for i in range(10):
                if self.current_heading is not None:
                    break
                else:
                    print("Aguardando inicializacao.")
                    time.sleep(0.5)
                heading_inicio=self.current_heading
                self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

                #print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

            for i in range(10):
                self.local_target_pub.publish(self.cur_target_pose)
                self.arm_state = self.arm()
                self.offboard_state = self.offboard()
                time.sleep(0.2)


            if self.takeoff_detection():
                print("Veiculo decolou!")
            else:
                print("Decolagem do veiculo falhou!")
                return

            '''
            decola e navega controlando velocidade a primeira vez
            '''
            obj_x0=0
            teta=self.current_heading*180/np.pi
            h=-34
            indicador=None
            log_altura=[]
            log_yaw=[]
            log_tempo=[]
            log_x=[]
            log_y=[]
            log_vel=[]

            primeira_deteccao=True
            contador=0
            yaw_compass=[]
            yaw_camera=[]
            yaw_kalman=[]
            ekf = DroneEKF()
            while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
                self.local_target_pub.publish(self.cur_target_pose)
                if self.camera_msg=='Parando Etapa 2!':

                    log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
                    log_etapa2 = pd.DataFrame(data=log_etapa2)
                    #log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)
                    log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa2.csv', index=False)

                    plt.clf()
                    plt.plot(log_tempo,log_yaw)
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Orientacao (º)')
                    plt.title('Orientacao x Tempo de Simulacao')
                    plt.grid()
                    #plt.savefig('log_OrientacaoxTempo_etapa2.png')

                    plt.clf()
                    fig=plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot3D(log_x,log_y,log_altura)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Altura (m)')
                    ax.set_title('Orientacao x Tempo de Simulacao')
                    #fig.savefig('log_posicaoDrone_etapa2.png')

                    break
                else:
                    centrox=self.camera_msg[0]/2
                    obj_x=self.camera_msg[2]
                    tolerancia=0.2*centrox
                    #print(teta)
                    if obj_x0 != obj_x:
                        if abs(centrox-obj_x)>tolerancia:
                            if obj_x>centrox:
                                print('Destino a direita')
                                indicador='D'
                                yaw_compass.append(teta)
                                if primeira_deteccao == True:
                                    primeira_deteccao = False
                                    tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
                                    if tetad<-180:
                                        tetad=tetad+360
                                elif primeira_deteccao == False:
                                    tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
                                    if tetad<-180:
                                        tetad=tetad+360
                                yaw_camera.append(tetad)
                                Tk=(yaw_compass[-1], yaw_camera[-1])
                                dados_kalman=ekf.step(Tk)
                                tetak=dados_kalman[0]
                                yaw_kalman.append(tetak)
                                contador=contador+1
                                if contador>=10:
                                    tetaobj=tetak
                                else:
                                    tetaobj=tetad
                                print('######################################### NOVA ORIENTACAO:', tetaobj)
                            else:
                                print('Destino a esquerda')
                                indicador='E'
                                yaw_compass.append(teta)
                                if teta>0:
                                    if primeira_deteccao == True:
                                        primeira_deteccao = False
                                        tetad=teta+kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad>180:
                                            tetad=tetad-360
                                    elif primeira_deteccao == False:
                                        tetad=yaw_camera[-1]+kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad>180:
                                            tetad=tetad-360
                                    yaw_camera.append(tetad)
                                else:
                                    if primeira_deteccao == True:
                                        primeira_deteccao = False
                                        tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad<-180:
                                            tetad=tetad+360
                                    elif primeira_deteccao == False:
                                        tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad<-180:
                                            tetad=tetad+360
                                    yaw_camera.append(tetad)
                                Tk=(yaw_compass[-1], yaw_camera[-1])
                                dados_kalman=ekf.step(Tk)
                                tetak=dados_kalman[0]
                                yaw_kalman.append(tetak)
                                contador=contador+1
                                if contador>=10:
                                    tetaobj=tetak
                                else:
                                    tetaobj=tetad
                                print('######################################### NOVA ORIENTACAO:', tetad)
                            while abs(tetaobj-self.current_heading*180/np.pi)>=2 or abs(h-self.local_pose.pose.position.z)>=0.1:
                                self.cur_target_pose=self.controle_orientacao(tetaobj, h)
                                self.local_target_pub.publish(self.cur_target_pose)
                                time.sleep(0.2)
                                teta=self.current_heading*180/np.pi
                                print('Orientacao atual do drone:', teta)
                                print('Orientacao desejada:', tetaobj)
                                if indicador=='D':
                                    print('Corrigindo para a direita')
                                else:
                                    print('Corrigindo para a esquerda')
                    else:
                        while abs(h-self.local_pose.pose.position.z)>=0.1:
                            teta=self.current_heading*180/np.pi
                            print('Orientacao atual do drone:', teta)
                            self.cur_target_pose=self.controle_orientacao(teta, h)
                            self.local_target_pub.publish(self.cur_target_pose)
                            time.sleep(0.2)
                        pass

                    (orientacao, velx, vely) = self.calcula_velocidade_drone(vel, teta)
                    self.velocidade_drone = self.controle_velocidade(velx, vely)
                    self.setpoint_velocidade_pub.publish(self.velocidade_drone)
                    print('Orientacao atual do drone:', teta)
                    x=self.local_pose.pose.position.x
                    xd=self.destino_msg[0]
                    y=self.local_pose.pose.position.y
                    yd=self.destino_msg[1]
                    dist=np.sqrt(np.square(x-xd)+np.square(y-yd))
                    print('Distancia do alvo:', dist)
                    obj_x0=obj_x
                    time.sleep(0.2)

                    pos_z=self.local_pose.pose.position.z
                    pos_x=self.local_pose.pose.position.x
                    pos_y=self.local_pose.pose.position.y
                    # vel_x=self.leitura_velocidade_drone.twist.linear.x
                    # vel_y=self.leitura_velocidade_drone.twist.linear.y
                    # vel_drone=np.sqrt((vel_x**2)+(vel_y**2))
                    yaw_atual=self.current_heading*180/np.pi
                    # ohm=self.leitura_velocidade_drone.twist.angular.z
                    t=rospy.get_time()

                    log_altura.append(pos_z)
                    log_x.append(pos_x)
                    log_y.append(pos_y)
                    log_yaw.append(yaw_atual)
                    # log_vel.append(vel_drone)
                    log_tempo.append(t)

                    if dist<=1:

                        log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
                        log_etapa2 = pd.DataFrame(data=log_etapa2)
                        #log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)
                        log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa2.csv', index=False)

                        plt.clf()
                        plt.plot(log_tempo,log_yaw)
                        plt.xlabel('Tempo (s)')
                        plt.ylabel('Orientacao (º)')
                        plt.title('Orientacao x Tempo de Simulacao')
                        plt.grid()
                        #plt.savefig('log_OrientacaoxTempo_etapa2.png')

                        plt.clf()
                        fig=plt.figure()
                        ax = plt.axes(projection='3d')
                        ax.plot3D(log_x,log_y,log_altura)
                        ax.set_xlabel('X (m)')
                        ax.set_ylabel('Y (m)')
                        ax.set_zlabel('Altura (m)')
                        ax.set_title('Orientacao x Tempo de Simulacao')
                        #fig.savefig('log_posicaoDrone_etapa2.png')

                        break

        else:
            '''
            navega controlando velocidade
            '''
            obj_x0=0
            teta=self.current_heading*180/np.pi
            indicador=None
            h=-34
            log_altura=[]
            log_yaw=[]
            log_tempo=[]
            log_x=[]
            log_y=[]
            log_vel=[]

            primeira_deteccao=True
            contador=0
            yaw_compass=[]
            yaw_camera=[]
            yaw_kalman=[]
            ekf = DroneEKF()
            while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
                self.local_target_pub.publish(self.cur_target_pose)
                if self.camera_msg=='Parando Etapa 2!':

                    log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
                    log_etapa2 = pd.DataFrame(data=log_etapa2)
                    #log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)
                    log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa2.csv', index=False)

                    plt.clf()
                    plt.plot(log_tempo,log_yaw)
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Orientacao (º)')
                    plt.title('Orientacao x Tempo de Simulacao')
                    plt.grid()
                    #plt.savefig('log_OrientacaoxTempo_etapa2.png')

                    plt.clf()
                    fig=plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot3D(log_x,log_y,log_altura)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Altura (m)')
                    ax.set_title('Orientacao x Tempo de Simulacao')
                    #fig.savefig('log_posicaoDrone_etapa2.png')

                    break
                else:
                    centrox=self.camera_msg[0]/2
                    obj_x=self.camera_msg[2]
                    tolerancia=0.2*centrox
                    #print(teta)
                    if obj_x0 != obj_x:
                        if abs(centrox-obj_x)>tolerancia:
                            if obj_x>centrox:
                                print('Destino a direita')
                                indicador='D'
                                yaw_compass.append(teta)
                                if primeira_deteccao == True:
                                    primeira_deteccao = False
                                    tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
                                    if tetad<-180:
                                        tetad=tetad+360
                                elif primeira_deteccao == False:
                                    tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
                                    if tetad<-180:
                                        tetad=tetad+360
                                yaw_camera.append(tetad)
                                Tk=(yaw_compass[-1], yaw_camera[-1])
                                dados_kalman=ekf.step(Tk)
                                tetak=dados_kalman[0]
                                yaw_kalman.append(tetak)
                                contador=contador+1
                                if contador>=10:
                                    tetaobj=tetak
                                else:
                                    tetaobj=tetad
                                print('######################################### NOVA ORIENTACAO:', tetaobj)
                            else:
                                print('Destino a esquerda')
                                indicador='E'
                                yaw_compass.append(teta)
                                if teta>0:
                                    if primeira_deteccao == True:
                                        primeira_deteccao = False
                                        tetad=teta+kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad>180:
                                            tetad=tetad-360
                                    elif primeira_deteccao == False:
                                        tetad=yaw_camera[-1]+kp*(abs(centrox-obj_x)/(2*centrox))
                                        if tetad>180:
                                            tetad=tetad-360
                                    yaw_camera.append(tetad)
                                else:
                                    if primeira_deteccao == True:
                                        primeira_deteccao = False
                                        tetad=teta-kp*(abs(centrox-obj_x)/centrox)
                                        if tetad<-180:
                                            tetad=tetad+360
                                    elif primeira_deteccao == False:
                                        tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/centrox)
                                        if tetad<-180:
                                            tetad=tetad+360
                                    yaw_camera.append(tetad)
                                Tk=(yaw_compass[-1], yaw_camera[-1])
                                dados_kalman=ekf.step(Tk)
                                tetak=dados_kalman[0]
                                yaw_kalman.append(tetak)
                                contador=contador+1
                                if contador>=10:
                                    tetaobj=tetak
                                else:
                                    tetaobj=tetad
                                print('######################################### NOVA ORIENTACAO:', tetad)
                            while abs(tetaobj-self.current_heading*180/np.pi)>=2 or abs(h-self.local_pose.pose.position.z)>=0.1:
                                self.cur_target_pose=self.controle_orientacao(tetaobj, h)
                                self.local_target_pub.publish(self.cur_target_pose)
                                time.sleep(0.2)
                                teta=self.current_heading*180/np.pi
                                print('Orientacao atual do drone:', teta)
                                print('Orientacao desejada:', tetaobj)
                                if indicador=='D':
                                    print('Corrigindo para a direita')
                                else:
                                    print('Corrigindo para a esquerda')
                    else:
                        while abs(h-self.local_pose.pose.position.z)>=0.1:
                            teta=self.current_heading*180/np.pi
                            print('Orientacao atual do drone:', teta)
                            self.cur_target_pose=self.controle_orientacao(teta, h)
                            self.local_target_pub.publish(self.cur_target_pose)
                            time.sleep(0.2)
                        pass

                    (orientacao, velx, vely) = self.calcula_velocidade_drone(vel, teta)
                    self.velocidade_drone = self.controle_velocidade(velx, vely)
                    self.setpoint_velocidade_pub.publish(self.velocidade_drone)
                    print('Orientacao atual do drone:', teta)
                    x=self.local_pose.pose.position.x
                    xd=self.destino_msg[0]
                    y=self.local_pose.pose.position.y
                    yd=self.destino_msg[1]
                    dist=np.sqrt(np.square(x-xd)+np.square(y-yd))
                    print('Distancia do alvo:', dist)
                    obj_x0=obj_x
                    time.sleep(0.2)

                    pos_z=self.local_pose.pose.position.z
                    pos_x=self.local_pose.pose.position.x
                    pos_y=self.local_pose.pose.position.y
                    # vel_x=self.leitura_velocidade_drone.twist.linear.x
                    # vel_y=self.leitura_velocidade_drone.twist.linear.y
                    # vel_drone=np.sqrt((vel_x**2)+(vel_y**2))
                    yaw_atual=self.current_heading*180/np.pi
                    # ohm=self.leitura_velocidade_drone.twist.angular.z
                    t=rospy.get_time()

                    log_altura.append(pos_z)
                    log_x.append(pos_x)
                    log_y.append(pos_y)
                    log_yaw.append(yaw_atual)
                    # log_vel.append(vel_drone)
                    log_tempo.append(t)

                    if dist<=1:

                        log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
                        log_etapa2 = pd.DataFrame(data=log_etapa2)
                        #log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)
                        log_etapa2.to_csv('/home/gabryelsr/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa2.csv', index=False)

                        plt.clf()
                        plt.plot(log_tempo,log_yaw)
                        plt.xlabel('Tempo (s)')
                        plt.ylabel('Orientacao (º)')
                        plt.title('Orientacao x Tempo de Simulacao')
                        plt.grid()
                        #plt.savefig('log_OrientacaoxTempo_etapa2.png')

                        plt.clf()
                        fig=plt.figure()
                        ax = plt.axes(projection='3d')
                        ax.plot3D(log_x,log_y,log_altura)
                        ax.set_xlabel('X (m)')
                        ax.set_ylabel('Y (m)')
                        ax.set_zlabel('Altura (m)')
                        ax.set_title('Orientacao x Tempo de Simulacao')
                        #fig.savefig('log_posicaoDrone_etapa2.png')

                        break

centro_msg=None
def centro_callback(data):
	global centro_msg
	centro_msg=data.data

msg=None
def callback(data):
	global msg
	msg=data.data

cam_msg=None
def camera_callback(data):
    global cam_msg
    cam_msg=data.data

rospy.init_node("controlador_offboard")

if __name__ == '__main__':
    h=2.5
    v=1.0
    # while centro_msg is None:
    #     rospy.Subscriber("configura_centro", String, centro_callback)
    #     if centro_msg is not None:
    #         dados_centro=centro_msg
    # C = [float(coordenadas) for coordenadas in dados_centro.split(",")]
    # print(C[0])
    # print(C[1])

    while msg is None:
        rospy.Subscriber("configura_trajetoria", String, callback)
        if msg is not None:
            dados=msg
    dados=dados.replace("[","")
    dados=dados.replace("]","")
    dados = [float(item) for item in dados.split(",")]

    con = Px4Controller()
    tam=int(len(dados)/3)
    x=np.zeros(tam)
    y=np.zeros(tam)
    yaw=np.zeros(tam)
    for i in range(tam):
        x[i]=dados[i]
        y[i]=dados[i+tam]
        yaw[i]=dados[i+2*tam]
    x=x.tolist()
    y=y.tolist()
    yaw=yaw.tolist()
    pontos=[]
    for i in range(tam):
        pontos.append((x[i],y[i],h,np.float32(yaw[i])))
    print(pontos)

    executado_pt1=False
    executado_pt2=False
    decola=True
    if executado_pt1 is False:
        con.missao_pt1(pontos)
        executado_pt1=True
        print('Fim da Etapa 1')
        decola=False
    if executado_pt1 is True and executado_pt2 is False:
        con.missao_pt2(decola)
        executado_pt2=True
        print('Fim da Etapa 2')
        while executado_pt2 is True:
            h=1.5
            con.fixa(decola, h)
        #decola=False
        # while cam_msg is None:
        #   if decola==True:
        #       con.fixa(decola)
        #       decola=False
        #   rospy.Subscriber('dados_camera_kf', String, camera_callback)
        #   if cam_msg is not None:
        #       dados_cam=cam_msg
        #       dados_cam=dados_cam.replace("[","")
        #       dados_cam=dados_cam.replace("]","")
        #       dados_cam = [float(item_cam) for item_cam in dados_cam.split(",")]
        #       #con.missao_pt2(v, dados_cam[0], dados_cam[2], decola)
        #       #print(dados_cam[0])
        #       #print(dados_cam[2])
        # cam_msg=None
