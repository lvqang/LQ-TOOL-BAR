import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib import rcParams



class AgentSimulator:

    def __init__(self):

        self.targets = [self.MovingTarget(i) for i in range(num_targets)]
        self.nodes = [self.NetworkNode(i) for i in range(num_nodes)]
        self.time = 0  # 当前仿真时间

    class MyCar:

        def __init__(self, lat0=23.108, lon0=113.2647, h0=10):
            self.lat0 = lat0
            self.lon0 = lon0
            self.h0   = h0
            a = 6378137.0  # WGS84地球长半轴（米）
            e2 = 0.00669437999014  # WGS84第一偏心率平方
            self.N = a / np.sqrt(1 - e2 * np.sin(lat0) ^ 2)  # 卯酉圈曲率半径


            self.car_pos = np.array(
                [np.random.uniform(-100, 100), np.random.uniform(-100, 100)]
            )
            self.vel = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
            self.acc = np.array([0, 0])
            self.motion_mode = "constant_velocity"  # 初始运动模式
            self.mode_switch_prob = 0.02  # 每步切换运动模式的概率
            self.last_switch_time = 0
            self.time = 0

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
            return [self.pos_N,self.pos_E,self.pos_U]

    class NetworkNode:
        def __init__(self):
            self.pos = np.array(
                [np.random.uniform(-150, 150), np.random.uniform(-150, 150)]
            )
            # 传递当前节点给传感器
            self.sensors = {
                "radar": self.RadarSensor(self),
                "irst": self.IRSensor(self),
            }
            self.filters = {}  # 每个目标一个滤波器
            self.comms = DataLinkTransceiver()

        def init_filters(self, targets):
            """初始化目标滤波器"""
            for target in targets:
                self.filters[target.id] = {
                    "radar": ExtendedKalmanFilter(target.id, sensor_type="radar"),
                    "irst": ExtendedKalmanFilter(target.id, sensor_type="irst"),
                    "fused": ExtendedKalmanFilter(target.id, sensor_type="fused"),
                }

        def sense_and_filter(self, target, dt):
            """对目标进行感知并滤波"""
            target_id = target.id
            measurements = {}

            # 生成各传感器测量值
            for sensor_type, sensor in self.sensors.items():
                if sensor.detect(target.pos):
                    measurements[sensor_type] = sensor.measure(target.pos)

            # 对每个传感器进行滤波
            for sensor_type in measurements:
                filter_ = self.filters[target_id][sensor_type]
                filter_.predict(dt)
                filter_.update(measurements[sensor_type], self.pos)  # 传递传感器位置

            # 融合所有可用传感器数据（改进的融合逻辑）
            if len(measurements) > 1:  # 仅当有多个传感器时融合
                fused_filter = self.filters[target_id]["fused"]
                fused_filter.predict(dt)
                fused_filter.fuse_measurements(measurements, self.pos)

            return measurements


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
                self.G1 = [q0**2+q1**2+q2**2+q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)]   #惯性系->机体系
                self.G2 = [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)]  # 惯性系->机体系
                self.G3 = [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]  # 惯性系->机体系
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
                self.gravityZ = 1-2*(q1**2+q2**2)

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



                return np.array([IMU_YAW, IMU_PITCH, IMU_ROLL])


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































# 主程序入口
if __name__ == "__main__":
    # 创建战场模拟器
    simulator = BattlefieldSimulator(num_targets=1, num_nodes=1)

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