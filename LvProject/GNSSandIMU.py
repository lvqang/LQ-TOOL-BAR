import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib import rcParams



class AgentSimulator:

    def __init__(self, lat, lon):
        lat0 = lat
        lon0 = lon
        self.mycar = self.MyCar(lat0,lon0)
        self.mynw =  self.NetworkNode()
        self.mykalm = self.ExtendedKalmanFilter()


    class MyCar:

        def __init__(self, lat0=23.108, lon0=113.2647, h0=10):
            self.lat0 = lat0
            self.lon0 = lon0
            self.h0   = h0
            a = 6378137.0  # WGS84地球长半轴（米）
            e2 = 0.00669437999014  # WGS84第一偏心率平方
            self.N = a / np.sqrt(1 - e2 * np.sin(lat0) ^ 2)  # 卯酉圈曲率半径

        def carUpdate(self, next_lat, next_lon, next_h=10, higt_fixed=1, dt=0.1):

            if higt_fixed==1:
                h = self.h0
            else:
                h=next_h
            # 北方向（N）
            self.pos_N = (next_lat - self.lat0) * (self.N + self.h0)
            # 东方向（E）
            self.pos_E = (next_lon - self.lon0) * (self.N + self.h0) * cos(self.lat0)
            # 天方向（U）
            self.pos_U = h - self.h0
            return [self.pos_N,self.pos_E]

    class ExtendedKalmanFilter:
        def __init__(self, sta, x, y):
            self.dt = 0.1  # 时间步长

            # 状态向量: [x, y, vx, vy, ax, ay]
            self.x = np.zeros(6)  # 初始状态  一维数组全0
            if(sta==1):#定位有效
                self.x[0]=x
                self.x[1] = y

            self.P = np.eye(6) * 1000  # 初始协方差矩阵  生成单位矩阵

            # 状态转移矩阵
            self.A = np.array(
                [
                    [1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
                    [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )

            self.B = np.array(
                [
                    [0.5 * self.dt ** 2, 0, 0, 0, 0, 0],
                    [0, 0.5 * self.dt ** 2, 0, 0, 0, 0],
                    [0, 0, self.dt, 0, 0, 0],
                    [0, 0, 0, self.dt, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )

            # 过程噪声协方差
            self.Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.001, 0.001])

            # 观测噪声协方差
            self.R = np.diag([10 ** 2, 10 ** 2, 20 ** 2])  # 东北天

        def predict(self, dt, accx,accy):
            """预测步骤"""
            if dt != self.dt:
                self.dt = dt
                self.A = np.array(
                    [
                        [1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
                        [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
                        [0, 0, 1, 0, self.dt, 0],
                        [0, 0, 0, 1, 0, self.dt],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )

            self.u = np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [accx],
                    [accy],
                ]
            )
            self.x = self.A @ self.x + self.B @ self.u
            self.P = self.A @ self.P @ self.A.T + self.Q

            return self.x.copy()

        def update(self,x,y):
            #观测值
            H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
            z_pred = H @ self.x
            z = np.array([
                    [x],
                    [y],
                    [0],
                    [0],
                    [0],
                    [0],])
            # 计算残差
            y = z - z_pred

            # 计算卡尔曼增益
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)

            # 更新状态和协方差
            self.x += K @ y
            self.P = (np.eye(6) - K @ H) @ self.P

            return self.x.copy()

        def fuse_measurements(self, measurements, sensor_pos):
            """融合来自多个传感器的测量（改进的融合逻辑）"""
            # 先进行标准预测
            # self.predict(self.dt)

            # 计算融合后的测量值（简化的加权平均）
            z_fused = np.zeros(2)  # 2×1
            R_fused_inv = np.zeros((2, 2))  # 2×2

            for sensor_type, z in measurements.items():
                if sensor_type == "radar":
                    R = np.diag([10 ** 2, np.radians(0.5) ** 2])
                elif sensor_type == "irst":
                    R = np.diag([50 ** 2, np.radians(1) ** 2])  # R xy=J⋅ Rrθ⋅ JT注意此处是否也需要对R进求偏导得到雅可比J
                else:
                    continue

                R_inv = np.linalg.inv(R)
                R_fused_inv += R_inv

                # 转换到全局坐标系
                if sensor_type in ["radar", "irst"]:
                    z_global = self._polar_to_cartesian(z, sensor_pos)
                else:
                    z_global = z

                z_fused += R_inv @ z_global

            # 计算融合后的测量和协方差
            R_fused = np.linalg.inv(R_fused_inv)
            z_fused = R_fused @ z_fused

            # 使用融合后的测量更新滤波器
            H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])  # 线性观测矩阵
            z_pred = H @ self.x
            y = z_fused - z_pred

            S = H @ self.P @ H.T + R_fused
            K = self.P @ H.T @ np.linalg.inv(S)

            self.x += K @ y
            self.P = (np.eye(6) - K @ H) @ self.P

        def _h(self, x, sensor_pos):
            """观测函数（状态到测量的映射）"""
            if self.sensor_type in ["radar", "irst"]:
                dx = x[0] - sensor_pos[0]
                dy = x[1] - sensor_pos[1]
                # 添加小量避免除零
                dx = max(dx, 1e-10)
                dy = max(dy, 1e-10)
                r = np.sqrt(dx ** 2 + dy ** 2)
                theta = np.arctan2(dy, dx)
                return np.array([r, theta])
            else:  # 融合后直接观测位置
                return np.array([x[0], x[1]])

        def _calculate_jacobian(self, x, sensor_pos):
            """计算观测函数的雅可比矩阵"""
            if self.sensor_type in ["radar", "irst"]:
                dx = x[0] - sensor_pos[0]
                dy = x[1] - sensor_pos[1]
                # 添加小量避免除零
                r_squared = dx ** 2 + dy ** 2 + 1e-10
                r = np.sqrt(r_squared)

                # 雅可比矩阵  非线性测量矩阵H需要进行对状态变量求偏导hx={{(dx**2+dy**2)**0.5,0,0...},{arctan2(dy,dx),0,0...}}
                H = np.array(
                    [
                        [dx / r, dy / r, 0, 0, 0, 0],
                        [-dy / r_squared, dx / r_squared, 0, 0, 0, 0],
                    ]
                )
                return H
            else:  # 线性观测矩阵
                return np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        def _polar_to_cartesian(self, z, sensor_pos):
            """将极坐标测量转换为笛卡尔坐标"""
            r, theta = z
            x = sensor_pos[0] + r * np.cos(theta)
            y = sensor_pos[1] + r * np.sin(theta)
            return np.array([x, y])

    class NetworkNode:
        def __init__(self):
            myQua = IMUQuaternion()

        class IMUQuaternion:
            def __init__(self, inist = 0):
                self.sensors = \
                {
                    "G-sensor":     self.Gsensor(),
                    "gyroscope":    self.Gyro(),
                    "magnetometer": self.Magne(),
                }
                ax = self.sensors["gyroscope"].mea_x#必须是机体系下的数据
                ay = self.sensors["gyroscope"].mea_y
                az = self.sensors["gyroscope"].mea_z
                yaw = np.degrees( np.arctan2(ay,ax))
                pitch=np.degrees( np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
                roll = np.degrees( np.arctan2(ay,az))
                #计算四元数
                cr = np.cos(roll / 2.0)
                sr = np.sin(roll / 2.0)
                cp = np.cos(pitch / 2.0)
                sp = np.sin(pitch / 2.0)
                cy = np.cos(yaw / 2.0)
                sy = np.sin(yaw / 2.0)
                if(inist==1):#（车辆静止、水平、朝向正北 / 初始航向）
                    self.q0 = cr * cp * cy + sr * sp * sy
                    self.q1 = sr * cp * cy - cr * sp * sy
                    self.q2 = cr * sp * cy + sr * cp * sy
                    self.q3 = cr * cp * sy - sr * sp * cy
                else:
                    self.q0 = 1
                    self.q1 = 0
                    self.q2 = 0
                    self.q3 = 0
                q0 = self.q0
                q1 = self.q1
                q2 = self.q2
                q3 = self.q3
                self.G[3][3] ={ {q0**2+q1**2+q2**2+q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)}   #惯性系->机体系
                                {2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)}  # 惯性系->机体系
                                {2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系
                self.GT[3][3]={ {q0**2+q1**2+q2**2+q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)}   #惯性系->机体系
                                {2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)}  # 惯性系->机体系
                                {2*(q1*q3-q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系
                self.gravityX = 0
                self.gravityY = 0
                self.gravityZ = 0
                self.AccX = 0
                self.AccY = 0
                self.AccZ = 0

                self.LpfX = 0
                self.LpfY = 0
                self.LpfZ = 0

                self.InteX = 0
                self.InteY = 0
                self.InteZ = 0

                self.deltaT = 0.1#s
                self.halftime = self.deltaT/2
                #磁力计一阶低通滤波
                self.Mag_lpX = 0
                self.Mag_lpY = 0
                self.Mag_lpZ = 0

            def QuaternionCal(self):
                q0 = self.q0
                q1 = self.q1
                q2 = self.q2
                q3 = self.q3
                self.gravityX = 2*(q1*q3-q0*q2)
                self.gravityY = 2*(q2*q3-q0*q1)
                self.gravityZ = 2*(q1**2+q2**2)-1#应该是-1

                Acc = self.sensors["G-sensor"].measure()
                self.AccX = Acc[0]
                self.AccY = Acc[1]
                self.AccZ = Acc[2]
                Gyro = self.sensors["gyroscope"].measure()
                self.GyroX = Gyro[0]
                self.GyroY = Gyro[1]
                self.GyroZ = Gyro[2]
                Mag = self.sensors["magnetometer"].measure()
                self.MagX = Mag[0]
                self.MagY = Mag[1]
                self.MagZ = Mag[2]

                norqul = np.sqrt(Acc[0]**2+Acc[1]**2+Acc[2]**2)
                if norqul>1e-6:
                    self.AccX /= norqul#归一化
                    self.AccY /= norqul
                    self.AccZ /= norqul
                else:
                    self.AccX = 0
                    self.AccY = 0
                    self.AccZ = 1
                #叉积法获取角误差
                errAccX = self.AccY*self.gravityZ-self.AccZ*self.gravityY#Acc.gravity.sinθ
                errAccY = self.AccZ*self.gravityX-self.AccX*self.gravityZ#Acc.gravity都是模长为1
                errAccZ = self.AccX*self.gravityY-self.AccY*self.gravityX#所以上述== sinθ≈θ
                #低通滤波器
                self.LpfX += 6.28*self.halftime*(errAccX-self.LpfX)
                self.LpfY += 6.28*self.halftime*(errAccY-self.LpfY)
                self.LpfZ += 6.28*self.halftime*(errAccZ-self.LpfZ)

                IMU_LIM = 0.034906  #积分限幅  2°
                ANGLE   = 0.017453  #°转为弧度
                DEG_ANG = 57.29578  #弧度转为°
                #计算积分环节
                Ki = 1
                Kp = 1
                yawcorrect = 1
                self.InteX += self.LpfX * Ki * 2 * self.halftime
                self.InteY += self.LpfY * Ki * 2 * self.halftime
                self.InteZ += self.LpfZ * Ki * 2 * self.halftime
                #积分限幅
                self.InteX = max(-IMU_LIM, min(IMU_LIM, self.InteX))
                self.InteY = max(-IMU_LIM, min(IMU_LIM, self.InteY))
                self.InteZ = max(-IMU_LIM, min(IMU_LIM, self.InteZ))
                #PID  通过加速度计在x轴 y轴的分量 校正陀螺仪的数据  同时还需先抵消陀螺仪偏航带来的干扰
                GloX = (self.GyroX - self.gravityX * yawcorrect) * ANGLE + (Kp * (self.LpfX + self.InteX))
                GloY = (self.GyroY - self.gravityY * yawcorrect) * ANGLE + (Kp * (self.LpfY + self.InteY))
                GloZ = (self.GyroZ - self.gravityZ * yawcorrect) * ANGLE

                #迭代四元数
                self.q0 = q0+(-q1*GloX - q2*GloY - q3*GloZ)*self.halftime
                self.q1 = q1+( q0*GloX + q2*GloZ - q3*GloY)*self.halftime
                self.q2 = q2+( q0*GloY - q1*GloZ + q3*GloX)*self.halftime
                self.q3 = q3+( q0*GloZ + q1*GloY - q3*GloX)*self.halftime
                #归一化
                norqul = np.sqrt(self.q0**2+self.q1**2+self.q2**2+self.q3**2)
                self.q0=self.q0/norqul
                self.q1=self.q1/norqul
                self.q2=self.q2/norqul
                self.q3=self.q3/norqul

                #计算欧拉角
                EndAngle0 = 2 * (self.q0 * self.q1 + self.q2 * self.q3)
                EndAngle1 = 1 - 2 * (self.q1**2 + self.q2**2)
                EndAngle2 = 2 * (self.q1 * self.q3 - self.q0 * self.q2)
                EndAngle3 = 2 * (-self.q1 * self.q2 - self.q0 * self.q3)
                EndAngle4 = 2 * (self.q0**2 + self.q1**2)-1
                IMU_YAW = np.atan2(EndAngle3,EndAngle4)*DEG_ANG
                IMU_PITCH=np.asin(EndAngle2)*DEG_ANG
                IMU_ROLL= np.atan2(EndAngle0, EndAngle1) * DEG_ANG

                #添加磁力计限航
                norqul = np.sqrt(self.MagX**2+self.MagY**2+self.MagZ**2)
                self.MagX = self.MagX / norqul
                self.MagY = self.MagY / norqul
                self.MagZ = self.MagZ / norqul
                self.Mag_lpX += 6.28 * 40 * self.halftime * (self.MagX - self.Mag_lpX)
                self.Mag_lpY += 6.28 * 40 * self.halftime * (self.MagY - self.Mag_lpY)
                self.Mag_lpZ += 6.28 * 40 * self.halftime * (self.MagZ - self.Mag_lpZ)
                #机体系转为惯性系
                # norqul = np.sqrt(self.gravityX**2+self.gravityY**2+self.gravityZ**2)
                # graX = self.gravityX / norqul
                # graY = self.gravityY / norqul
                # graZ = self.gravityZ / norqul
                # norqul = np.sqrt(self.Mag_lpX ** 2 + self.Mag_lpY ** 2 + self.Mag_lpZ ** 2)
                # MagX = self.Mag_lpX / norqul
                # MagY = self.Mag_lpY / norqul
                # MagZ = self.Mag_lpZ / norqul
                # gx=0
                # gy=0
                # gz=-1
                q0 = self.q0
                q1 = self.q1
                q2 = self.q2
                q3 = self.q3
                self.G[3][3] ={ {q0**2+q1**2+q2**2+q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)}   #惯性系->机体系
                                {2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)}  # 惯性系->机体系
                                {2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系
                self.GT[3][3]={ {q0**2+q1**2+q2**2+q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)}   #惯性系->机体系
                                {2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)}  # 惯性系->机体系
                                {2*(q1*q3-q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系

                MagX = self.Mag_lpX
                MagY = self.Mag_lpY
                MagZ = self.Mag_lpZ
                MagProX = self.GT[0][0]*MagX+self.GT[0][1]*MagY+self.GT[0][2]*MagZ
                MagProY = self.GT[1][0]*MagX+self.GT[1][1]*MagY+self.GT[1][2]*MagZ
                MagProZ = self.GT[2][0]*MagX+self.GT[2][1]*MagY+self.GT[2][2]*MagZ
                norqul = np.sqrt(MagProX ** 2 + MagProY ** 2 + MagProZ ** 2)

                yaw_correct = IMU_YAW
                if MagProX!=0 and MagProY!=0 and MagProZ!=0:
                    yaw_mag = np.atan2(MagProY/norqul,MagProX/norqul)*DEG_ANG
                    yaw_correct = Kp*0.8* self.To_180_degrees(yaw_mag - IMU_YAW)

                return np.array([yaw_correct, IMU_PITCH, IMU_ROLL])

            def To_180_degrees(self, x):
                k=0
                if(x>180):
                    k=x-180
                elif(x<-180):
                    k=x+180
                return k
            class Gsensor:
                def __init__(self):
                    self.miu_x = 0
                    self.miu_y = 0
                    self.miu_z = 1  # 惯性系下

                    self.sigma = 0.01  # 单位g

                def measure(self):
                    self.mea_x = self.miu_x + np.random.normal(0, self.sigma)
                    self.mea_y = self.miu_y + np.random.normal(0, self.sigma)
                    self.mea_z = self.miu_z + np.random.normal(0, self.sigma)  # 机体系
                    return np.array([self.mea_x, self.mea_y, self.mea_z])

            class Gyro:
                def __init__(self):
                    self.miu_psi = 0.0
                    self.miu_sita = 0.0
                    self.miu_phy = 0.0  # 机体系
                    self.dt = 0

                    self.sigma = 0.0002  # rad/s

                def measure(self):
                    self.miu_sita = self.miu_sita + np.random.normal(0, self.sigma)
                    self.miu_phy = self.miu_phy + np.random.normal(0, self.sigma)
                    self.miu_psi = self.miu_psi + self.detect(self.dt) + np.random.normal(0, self.sigma)
                    self.dt += 0.1

                def detect(self, dt):
                    auto = 0.0
                    if dt < 2:
                        auto = np.radians(20)  # 右转20°
                    elif dt >= 2 and dt <= 4:
                        auto = np.radians(-20)  # 左转20°
                    else:
                        auto = 0  # 直行
                    return auto

            class Magne:
                def __init__(self):
                    self.mx = 0.0#惯性系下的
                    self.my = 45.0#数据自己定义 可以根据实际情况调整
                    self.mz = 0.0

                    self.msigma = 0.1
                    self.dt = 0
                def measure(self):
                    self.mx = self.mx + np.random.normal(0, self.msigma) + self.detect(self, self.dt)
                    self.my = self.my + np.random.normal(0, self.msigma) - self.detect(self, self.dt)
                    self.mz = self.mz + np.random.normal(0, self.msigma)
                    self.dt += 0.1
                def detect(self, dt):
                    auto = 0.0
                    if dt < 2:
                        auto = 10.0
                    elif dt >= 2 and dt <= 4:
                        auto = -10.0
                    else:
                        auto = 0  # 直行
                    return auto


class BattlefieldVisualizer:

    def __init__(self, simulator):
        self.sim = simulator
        self.fig, (self.ax, self.error_ax) = plt.subplots(1, 2, figsize=(18, 8))
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        self.ax.set_xlabel("X坐标 (m)")
        self.ax.set_ylabel("Y坐标 (m)")
        self.ax.set_title("卡尔曼传感器融合模拟")

        # 初始化绘图元素
        # self.mathpos = self.ax.scatter(
        #     [], [], c="red", marker="o", s=50, label="MATH"
        # )
        # self.gpspos = self.ax.scatter(
        #     [], [], c="black", marker="x", s=100, label="GPS"
        # )
        # self.kalmpos = self.ax.scatter(
        #     [], [], c="blue", marker="|", s=150, label="Kalm"
        # )
        (self.gpspos,) = self.ax.plot(
            [], [], "k.", markersize=4, alpha=0.3, label="GPS定位"
        )
        (self.mathpos,) = self.ax.plot(
            [], [], "r.", markersize=4, alpha=0.3, label="动力学预测"
        )
        (self.kalmpos,) = self.ax.plot(
            [], [], "b.", markersize=4, alpha=0.3, label="卡尔曼融合"
        )

        # 误差统计图表
        self.error_ax.set_title("误差")
        self.error_ax.set_xlabel("时间步")
        self.error_ax.set_ylabel("误差 (m)")

        (self.math_gps,) = self.ax.plot(
            [], [], "k.", markersize=4, alpha=0.3, label="MATH-GPS"
        )
        (self.kalm_gps,) = self.ax.plot(
            [], [], "r.", markersize=4, alpha=0.3, label="KALM-GPS"
        )



        # 历史数据存储
        self.gps_pos = []
        self.math_pos = []
        self.kalm_pos = []

        self.math_gps_pos = []
        self.kalm_gps_pos = []

        # 图例
        self.ax.legend(loc="upper right", fontsize=8)
        self.error_ax.legend(loc="upper right", fontsize=8)

    def update(self, frame):
        """动画更新函数"""
        self.poscont += 1
        if self.poscont >= self.poslen:
            return
        # 更新所有目标
        sim = self.sim
        #获取相对起始点的位移
        carpos = sim.mycar.carUpdate(pos[self.poscont][0],pos[self.poscont][1])
        self.gps_pos.append(carpos)

        #获取偏航角
        carqua = sim.mynw.myQua.QuaternionCal()




        # 更新绘图数据
        true_positions = np.array([t.pos for t in self.sim.targets])
        node_positions = np.array([n.pos for n in self.sim.nodes])

        # 更新散点图
        self.targets_scat.set_offsets(true_positions)
        self.nodes_scat.set_offsets(node_positions)

        # 更新测量点
        if radar_measurements:
            radar_pts = np.array(radar_measurements)
            self.radar_meas.set_data(radar_pts[:, 0], radar_pts[:, 1])

        if irst_measurements:
            irst_pts = np.array(irst_measurements)
            self.irst_meas.set_data(irst_pts[:, 0], irst_pts[:, 1])

        # 更新轨迹
        for target_id, target in enumerate(self.sim.targets):
            self.true_pos_history[target_id].append(target.pos)

            # 从第一个节点获取滤波结果
            node = self.sim.nodes[0]
            radar_filter = node.filters[target_id]["radar"]
            irst_filter = node.filters[target_id]["irst"]
            fused_filter = node.filters[target_id]["fused"]

            # 预测所有滤波器状态（即使没有新测量）
            # radar_filter.predict(0.1)
            # irst_filter.predict(0.1)
            # fused_filter.predict(0.1)

            # 存储估计位置
            radar_pos = radar_filter.x[:2]
            irst_pos = irst_filter.x[:2]
            fused_pos = fused_filter.x[:2]

            self.radar_pos_history[target_id].append(radar_pos)
            self.irst_pos_history[target_id].append(irst_pos)
            self.fused_pos_history[target_id].append(fused_pos)

            # 计算误差
            radar_error = np.linalg.norm(target.pos - radar_pos)
            irst_error = np.linalg.norm(target.pos - irst_pos)
            fused_error = np.linalg.norm(target.pos - fused_pos)

            self.radar_error_history[target_id].append(radar_error)
            self.irst_error_history[target_id].append(irst_error)
            self.fused_error_history[target_id].append(fused_error)

            # 更新轨迹线
            self.true_trajs[target_id].set_data(
                [pos[0] for pos in self.true_pos_history[target_id]],
                [pos[1] for pos in self.true_pos_history[target_id]],
            )

            self.radar_trajs[target_id].set_data(
                [pos[0] for pos in self.radar_pos_history[target_id]],
                [pos[1] for pos in self.radar_pos_history[target_id]],
            )

            self.irst_trajs[target_id].set_data(
                [pos[0] for pos in self.irst_pos_history[target_id]],
                [pos[1] for pos in self.irst_pos_history[target_id]],
            )

            self.fused_trajs[target_id].set_data(
                [pos[0] for pos in self.fused_pos_history[target_id]],
                [pos[1] for pos in self.fused_pos_history[target_id]],
            )

            # 更新误差椭圆
            P = fused_filter.P
            eigenvalues, eigenvectors = np.linalg.eig(P[:2, :2])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)

            self.error_ellipses[target_id].center = fused_filter.x[0], fused_filter.x[1]
            self.error_ellipses[target_id].width = width
            self.error_ellipses[target_id].height = height
            self.error_ellipses[target_id].angle = angle

            # 更新误差图
            self.radar_error_lines[target_id].set_data(
                range(len(self.radar_error_history[target_id])),
                self.radar_error_history[target_id],
            )

            self.irst_error_lines[target_id].set_data(
                range(len(self.irst_error_history[target_id])),
                self.irst_error_history[target_id],
            )

            self.fused_error_lines[target_id].set_data(
                range(len(self.fused_error_history[target_id])),
                self.fused_error_history[target_id],
            )

        # 调整误差图坐标轴
        self.error_ax.relim()
        self.error_ax.autoscale_view()

        # 更新时间
        self.sim.time += 0.1

        # 返回所有需要重绘的对象
        return (
            (self.targets_scat, self.nodes_scat, self.radar_meas, self.irst_meas)
            + tuple(self.true_trajs)
            + tuple(self.radar_trajs)
            + tuple(self.irst_trajs)
            + tuple(self.fused_trajs)
            + tuple(self.error_ellipses)
            + tuple(self.radar_error_lines)
            + tuple(self.irst_error_lines)
            + tuple(self.fused_error_lines)
        )

    def calculate_accuracy_metrics(self):
        """计算并返回精度指标"""
        metrics = {}
        for target_id in range(self.sim.num_targets):
            radar_errors = self.radar_error_history[target_id]
            irst_errors = self.irst_error_history[target_id]
            fused_errors = self.fused_error_history[target_id]

            if not fused_errors:
                continue

            metrics[target_id] = {
                "radar": {
                    "rmse": np.sqrt(np.mean(np.square(radar_errors))),
                    "mae": np.mean(radar_errors),
                    "max": np.max(radar_errors),
                    "min": np.min(radar_errors),
                },
                "irst": {
                    "rmse": np.sqrt(np.mean(np.square(irst_errors))),
                    "mae": np.mean(irst_errors),
                    "max": np.max(irst_errors),
                    "min": np.min(irst_errors),
                },
                "fused": {
                    "rmse": np.sqrt(np.mean(np.square(fused_errors))),
                    "mae": np.mean(fused_errors),
                    "max": np.max(fused_errors),
                    "min": np.min(fused_errors),
                },
            }

        return metrics

    def print_accuracy_report(self):
        """打印精度报告"""
        metrics = self.calculate_accuracy_metrics()

        if not metrics:
            print("无有效误差数据")
            return

        print("\n===== 目标跟踪精度报告 =====")

        for target_id, target_metrics in metrics.items():
            print(f"\n目标 {target_id+1}:")

            for sensor_type, stats in target_metrics.items():
                sensor_name = {"radar": "雷达", "irst": "红外", "fused": "融合"}.get(
                    sensor_type, sensor_type
                )

                print(f"\n  {sensor_name} 传感器:")
                print(f"    RMSE: {stats['rmse']:.2f} m")
                print(f"    MAE:  {stats['mae']:.2f} m")
                print(f"    最大误差: {stats['max']:.2f} m")
                print(f"    最小误差: {stats['min']:.2f} m")

        # 计算总体平均误差
        if metrics:
            overall_radar_mae = np.mean([m["radar"]["mae"] for m in metrics.values()])
            overall_irst_mae = np.mean([m["irst"]["mae"] for m in metrics.values()])
            overall_fused_mae = np.mean([m["fused"]["mae"] for m in metrics.values()])

            print(f"\n===== 总体性能对比 =====")
            print(f"  雷达平均误差: {overall_radar_mae:.2f} m")
            print(f"  红外平均误差: {overall_irst_mae:.2f} m")
            print(f"  融合平均误差: {overall_fused_mae:.2f} m")
            if overall_radar_mae > 0:
                print(
                    f"  融合提升: {100 * (1 - overall_fused_mae / overall_radar_mae):.2f}% (相对于雷达)"
                )
            if overall_irst_mae > 0:
                print(
                    f"  融合提升: {100 * (1 - overall_fused_mae / overall_irst_mae):.2f}% (相对于红外)"
                )


# 主程序入口
if __name__ == "__main__":
    # 获取定位并存储  从日志中获取
    pos = []
    #......

    poslen = len(pos)
    poscont = 0

    simulator = AgentSimulator(pos[0][0],pos[0][1])

    # 创建可视化器
    visualizer = BattlefieldVisualizer(simulator)

    # 定义动画更新函数
    def animate(frame):
        return visualizer.update(frame)

    # 初始化动画
    ani = FuncAnimation(
        visualizer.fig,
        animate,#更新函数
        frames=np.arange(0, 200),#帧数
        interval=50,#dt=50ms
        blit=True,#只更新变化区域
        repeat=False,#是否重复
    )

    # 显示图例
    visualizer.ax.legend(loc="upper right", fontsize=6)
    visualizer.error_ax.legend(loc="upper right", fontsize=6)

    # 启动动画显示
    plt.tight_layout()
    plt.show()

    # 动画结束后输出精度报告
    visualizer.print_accuracy_report()