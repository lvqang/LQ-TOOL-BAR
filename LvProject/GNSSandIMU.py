import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib import rcParams

class AgentSimulator:

    def __init__(self, pos):
        lat0 = pos[0][1]
        lon0 = pos[0][0]
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
            self.N = a / np.sqrt(1 - e2 * np.sin(lat0)**2)  # 卯酉圈曲率半径

        def carUpdate(self, next_lat, next_lon, next_h=10, higt_fixed=1, dt=0.1):

            if higt_fixed==1:
                h = self.h0
            else:
                h=next_h
            # 北方向（N）
            self.pos_N = (next_lat - self.lat0) * (self.N + self.h0)
            # 东方向（E）
            self.pos_E = (next_lon - self.lon0) * (self.N + self.h0) * np.cos(self.lat0)
            # 天方向（U）
            self.pos_U = h - self.h0
            return [self.pos_N,self.pos_E]

    class ExtendedKalmanFilter:
        def __init__(self, sta=1):
            self.dt = 0.1  # 时间步长

            # 状态向量: [x, y, vx, vy, ax, ay]
            self.x = np.zeros(6)  # 初始状态  一维数组全0
            if(sta==1):#定位有效
                self.x[0]=0#初始位置就是0  
                self.x[1] = 0

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
                    [0, 0, 0, 0, 0.5 * self.dt ** 2, 0],
                    [0, 0, 0, 0, 0, 0.5 * self.dt ** 2],
                    [0, 0, 0, 0, self.dt, 0],
                    [0, 0, 0, 0, 0, self.dt],
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
            self.myQua = self.IMUQuaternion()

        class IMUQuaternion:
            def __init__(self, inist = 0):
                self.sensors = \
                {
                    "G-sensor":     self.Gsensor(),
                    "gyroscope":    self.Gyro(),
                    "magnetometer": self.Magne(),
                }
                ax = self.sensors["gyroscope"].miu_psi #必须是机体系下的数据
                ay = self.sensors["gyroscope"].miu_sita
                az = self.sensors["gyroscope"].miu_phy
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
                self.G =[ [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],   #惯性系->机体系
                          [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],  # 惯性系->机体系
                          [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2] ]  # 惯性系->机体系
                self.GT=[ [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],    #机体系->惯性系
                          [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],    # 机体系->惯性系
                          [2*(q1*q3-q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2] ]  # 机体系->惯性系
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
            def ENDtoBody(self, q,xyz):
                q0,q1,q2,q3 = q
                x,y,z=xyz
                self.G[3][3] =  {{q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)},   # 惯性系->机体系
                                 {2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)},   # 惯性系->机体系
                                 {2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系
                x1 = self.G[0][0]*x+self.G[0][1]*y+self.G[0][2]*z
                y1 = self.G[1][0]*x+self.G[1][1]*y+self.G[1][2]*z
                z1 = self.G[2][0]*x+self.G[2][1]*y+self.G[2][2]*z
                return [x1,y1,z1]
            def BodytoEND(self, q,xyz):
                q0,q1,q2,q3 = q
                x,y,z=xyz
                self.GT[3][3] = {{q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)},   # 惯性系->机体系
                                 {2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)},   # 惯性系->机体系
                                 {2*(q1*q3-q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2} }  # 惯性系->机体系
                x1 = self.GT[0][0]*x+self.GT[0][1]*y+self.GT[0][2]*z
                y1 = self.GT[1][0]*x+self.GT[1][1]*y+self.GT[1][2]*z
                z1 = self.GT[2][0]*x+self.GT[2][1]*y+self.GT[2][2]*z
                return [x1,y1,z1]


            def QuaternionCal(self):
                q0 = self.q0
                q1 = self.q1
                q2 = self.q2
                q3 = self.q3
                #重力加速度转为机体系
                self.gravityX = 2*(q1*q3-q0*q2)
                self.gravityY = 2*(q2*q3-q0*q1)
                self.gravityZ = 2*(q1**2+q2**2)-1#应该是-1

                Acc = self.sensors["G-sensor"].measure()
                self.AccX = Acc[0]
                self.AccY = Acc[1]
                self.AccZ = Acc[2]
                Gyro = self.sensors["gyroscope"].measure()#获取的是角速度
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
                    print("[G-sensor]:norqul = %f",norqul)
                    self.AccX = 0
                    self.AccY = 0
                    self.AccZ = 1
                #加速度计和重力分量四元数 叉积法获取角误差
                errAccX = self.AccY*self.gravityZ-self.AccZ*self.gravityY#Acc.gravity.sinθ
                errAccY = self.AccZ*self.gravityX-self.AccX*self.gravityZ#Acc.gravity都是模长为1
                errAccZ = self.AccX*self.gravityY-self.AccY*self.gravityX#所以上述== sinθ≈θ
                #低通滤波器
                f = 20#截止频率
                alpha = 2*3.14*f*self.deltaT/(1 + 2 * 3.14 * f * self.deltaT)
                self.LpfX =alpha*(errAccX) + (1-alpha)*self.LpfX
                self.LpfY = alpha * (errAccY) + (1 - alpha) * self.LpfY
                self.LpfZ =alpha*(errAccZ) + (1-alpha)*self.LpfZ

                # 添加磁力计限航
                # 磁力计转为机体系
                mag = self.ENDtoBody([q0,q1,q2,q3],[self.MagX,self.MagY,self.MagZ])
                magX,magY,magZ = mag
                # 惯性系和机体系下的叉积法获取角误差
                errmagX = self.MagY * magZ - self.MagZ * magY  #
                errmagY = self.MagZ * magX - self.MagX * magZ  #
                errmagZ = self.MagX * magY - self.MagY * magX  #
                # 低通滤波器
                f = 20  # 截止频率
                alpha = 2 * 3.14 * f * self.deltaT / (1 + 2 * 3.14 * f * self.deltaT)
                self.Mag_lpX = alpha * (errmagX) + (1 - alpha) * self.Mag_lpX
                self.Mag_lpY = alpha * (errmagY) + (1 - alpha) * self.Mag_lpY
                self.Mag_lpZ = alpha * (errmagZ) + (1 - alpha) * self.Mag_lpZ


                IMU_LIM = 0.034906  #积分限幅  2°
                ANGLE   = 0.017453  #°转为弧度
                DEG_ANG = 57.29578  #弧度转为°
                #计算积分环节
                Ki = 1
                Kp = 1
                Ki_z = 1
                Kp_z = 1
                yawcorrect = 0
                self.InteX += self.LpfX * Ki * self.deltaT
                self.InteY += self.LpfY * Ki * self.deltaT
                self.InteZ += self.Mag_lpZ * Ki * self.deltaT#偏航角用磁力计
                #积分限幅
                self.InteX = max(-IMU_LIM, min(IMU_LIM, self.InteX))
                self.InteY = max(-IMU_LIM, min(IMU_LIM, self.InteY))
                self.InteZ = max(-IMU_LIM, min(IMU_LIM, self.InteZ))
                #PID  通过加速度计在x轴 y轴的分量 校正陀螺仪的数据
                GloX = (self.GyroX - self.gravityX * yawcorrect) + (Kp  * (self.LpfX    + self.InteX))#rad/s
                GloY = (self.GyroY - self.gravityY * yawcorrect) + (Kp  * (self.LpfY    + self.InteY))
                GloZ = (self.GyroZ - self.gravityZ * yawcorrect) + (Kp_z* (self.Mag_lpZ + self.InteX))##偏航角用磁力计

                #迭代四元数dq=0.5* q * w  其中w是角速度  q(t+1)=q(t+1)+dq*dt
                self.q0 = q0+(-q1*GloX - q2*GloY - q3*GloZ)*self.halftime
                self.q1 = q1+( q0*GloX + q2*GloZ - q3*GloY)*self.halftime
                self.q2 = q2+( q0*GloY + q3*GloX - q1*GloZ)*self.halftime
                self.q3 = q3+( q0*GloZ + q1*GloY - q2*GloX)*self.halftime
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
                IMU_YAW = np.atan2(EndAngle3,EndAngle4)*DEG_ANG#第一个数分子  第二个是分母
                IMU_PITCH=np.asin(EndAngle2)*DEG_ANG
                IMU_ROLL= np.atan2(EndAngle0, EndAngle1) * DEG_ANG

                #计算加速度计在惯性系的数据
                Acc = self.BodytoEND(self, [self.q0,self.q1,self.q2,self.q3],[self.AccX,self.AccY,self.AccZ])
                AX,AY,AZ = Acc
                return [AX,AY,AZ]

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

    def __init__(self, simulator, pos):
        self.sim = simulator
        self.fig, (self.ax, self.error_ax) = plt.subplots(1, 2, figsize=(18, 8))
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        self.ax.set_xlabel("X坐标 (m)")
        self.ax.set_ylabel("Y坐标 (m)")
        self.ax.set_title("卡尔曼传感器融合模拟")
        self.poscont = 0
        self.poslen = len(pos)
        self.pos = pos

        self.targets_scat = self.ax.scatter(
            [], [], c="red", marker="o", s=50, label="目标"
        )
        # 更新散点图
        self.targets_scat.set_offsets(pos)

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

        (self.math_gps,) = self.error_ax.plot(
            [], [], "k.", markersize=4, alpha=0.3, label="MATH-GPS"
        )
        (self.kalm_gps,) = self.error_ax.plot(
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
        # if self.poscont >= self.poslen:
        #     return
        # 更新所有目标
        sim = self.sim
        #获取相对起始点的位移
        if len(self.pos)>1:
            carpos = sim.mycar.carUpdate(self.pos[self.poscont][0],self.pos[self.poscont][1])
            self.gps_pos.append(carpos)
        else:
            carpos = sim.mycar.carUpdate(self.pos[0][0], self.pos[0][1])
            self.gps_pos.append(carpos)

        #获取加速度
        acc = sim.mynw.myQua.QuaternionCal()

        #计算卡尔曼
        pre = sim.mykalm.predict(0.1, acc[0],acc[1])
        self.math_pos.append(pre)

        updata = sim.mykalm.update(carpos[0],carpos[1])
        self.kalm_pos.append(updata)

        #计算误差
        self.math_gps_pos.append(pre-carpos)
        self.kalm_gps_pos.append(updata-carpos)

        # 更新绘图数据
        self.gps_pos.set_data(carpos[:,0], carpos[:,1])
        self.math_pos.set_data(pre[:,0], pre[:,1])
        self.kalm_pos.set_data(updata[:,0], updata[:,1])

        self.math_gps_pos.set_data(pre[:,0]-carpos[:,0], pre[:,1]-carpos[:,1])
        self.kalm_gps_pos.set_data(updata[:,0]-carpos[:,0], updata[:,1]-carpos[:,1])

        # 返回所有需要重绘的对象
        return (self.targets_scat,self.gps_pos,self.math_pos,self.kalm_pos,self.math_gps_pos,self.kalm_gps_pos)

class ParseFile:
    def __init__(self):
        self.file_path = "D:/installSoftware/36_Pycharm/LQ-TOOL-BAR/LvProject/1soc_encrypt_20260202_113619decode.log"
        self.keyWord = "/usbAppGNSS"  # 需要过滤的关键词

    def parse_file_dialog(self):

        self.gpsBuffer = []
        self.gpsBufferDetail = []
        gpsBufferCell = {}
        self.pos = [[0,0]]
        self.k=0

        try:
            if (".log" not in str(self.file_path)):
                self.text_display.setText("!!!!not log file!!!!")
                return self.pos
            lines = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lines.append(line.strip())#读取每行截止到换行符
            for gpsline in lines :
                if(self.keyWord in gpsline):
                    self.k += 1
                    self.gpsBuffer.append(gpsline)
                    time = self.get_filter_str(gpsline, '', ' I/')  # 纬度
                    part111 = gpsline.split(self.keyWord, 1)#防止日志错乱
                    status = int(self.get_filter_str(part111[1], '): ', ' '))
                    loc = self.get_filter_str(part111[1], 'lon::', ' lat::')#经度
                    locNum = 0.0
                    latNum = 0.0
                    if(loc and loc.strip()):
                        locNum = float(loc)  # 经度
                    lat = self.get_filter_str(part111[1], 'lat::', ' ')  # 纬度
                    if (lat and lat.strip()):
                        latNum = float(lat)  # 经度
                    # print("错误:", time,status,locNum,latNum)
                    self.gpsBufferDetail.append({time,status,locNum,latNum})
                    if self.k==1:
                        self.pos[0]=[locNum, latNum]
                    else:
                        self.pos.appned([locNum,latNum])

            return self.pos
        except Exception as e:
            print("错误:",e)
            return self.pos
        #打印

    def get_filter_str(self,text, left,right):#分割解析数据
        if text is None:
            return ""
        # 转为字符串（防数字等类型）
        s = str(text)
        # 按第一个 ':' 分割，取第一部分
        if(left==''):
            part = s.split(right, 1)
            return part[0]
        part = s.split(left, 1)
        if(len(part)>1):
            curstr = part[1]
            part = curstr.split(right, 1)
            if(len(part)>1):
                return part[0]
            else:
                return ""
        else:
            return ""
        # return s.split('：', 1)[0].strip()# strip() 去除前后空格 2)[0].strip()  # strip() 去除前后空格  1表示分隔1次 [1]表示第二部分

# 主程序入口
if __name__ == "__main__":
    # 获取定位并存储  从日志中获取
    myfile = ParseFile()
    #......
    pos = myfile.parse_file_dialog()

    simulator = AgentSimulator(pos)

    # 创建可视化器
    visualizer = BattlefieldVisualizer(simulator,pos)

    # 定义动画更新函数
    def animate(frame):
        return visualizer.update(frame)

    # 初始化动画
    ani = FuncAnimation(
        visualizer.fig,
        animate,#更新函数
        frames=np.arange(0, len(pos)),#帧数
        interval=100,#dt=50ms
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