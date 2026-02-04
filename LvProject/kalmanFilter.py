import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib import rcParams

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False


class BattlefieldSimulator:
    """战场环境模拟器：管理目标、传感器节点和数据链通信"""

    def __init__(self, num_targets=3, num_nodes=5):
        self.num_targets = num_targets
        self.num_nodes = num_nodes
        self.targets = [self.MovingTarget(i) for i in range(num_targets)]
        self.nodes = [self.NetworkNode(i) for i in range(num_nodes)]
        self.time = 0  # 当前仿真时间

    class MovingTarget:
        """目标运动模型：支持匀加速运动和机动切换"""

        def __init__(self, target_id):
            self.id = target_id
            self.pos = np.array(
                [np.random.uniform(-100, 100), np.random.uniform(-100, 100)]
            )
            self.vel = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
            self.acc = np.array([0, 0])
            self.motion_mode = "constant_velocity"  # 初始运动模式
            self.mode_switch_prob = 0.02  # 每步切换运动模式的概率
            self.last_switch_time = 0
            self.time = 0

        def update(self, dt=0.1):
            """更新目标状态，支持运动模式随机切换"""
            if (
                np.random.random() < self.mode_switch_prob
                and self.time - self.last_switch_time > 10
            ):
                self.switch_motion_mode()
                self.last_switch_time = self.time

            if self.motion_mode == "constant_velocity":
                self.acc = np.random.normal(0, 0.1, 2)
            elif self.motion_mode == "constant_acceleration":
                self.acc = np.random.normal(0, 0.5, 2)
            elif self.motion_mode == "maneuvering":
                t = self.time
                self.acc = np.array([2 * np.sin(t / 10), 2 * np.cos(t / 10)])

            self.vel += self.acc * dt
            self.pos += self.vel * dt
            self.time += dt

        def switch_motion_mode(self):
            """随机切换运动模式"""
            modes = ["constant_velocity", "constant_acceleration", "maneuvering"]
            current_idx = modes.index(self.motion_mode)
            new_idx = (current_idx + np.random.randint(1, 3)) % 3
            self.motion_mode = modes[new_idx]
            print(f"目标 {self.id} 切换到 {self.motion_mode} 模式")

    class NetworkNode:
        """传感器节点：包含多种传感器和分布式卡尔曼滤波器"""

        def __init__(self, node_id):
            self.id = node_id
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

        class RadarSensor:
            """雷达传感器模型"""

            def __init__(self, node):
                self.node = node  # 保存对父节点的引用
                self.max_range = 200
                self.azimuth_range = np.radians(120)
                self.range_noise = 10
                self.angle_noise = np.radians(0.5)
                self.detection_prob = 0.95

            def detect(self, target_pos):
                """判断是否能检测到目标"""
                dx = target_pos[0] - self.node.pos[0]  # 使用节点位置
                dy = target_pos[1] - self.node.pos[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance > self.max_range:
                    return False

                angle = np.arctan2(dy, dx)
                if abs(angle) > self.azimuth_range / 2:
                    return False

                return np.random.random() < self.detection_prob

            def measure(self, target_pos):
                """生成雷达测量值（极坐标）"""
                dx = target_pos[0] - self.node.pos[0]  # 使用节点位置
                dy = target_pos[1] - self.node.pos[1]
                true_range = np.sqrt(dx**2 + dy**2)
                true_angle = np.arctan2(dy, dx)

                measured_range = true_range + np.random.normal(0, self.range_noise)
                measured_angle = true_angle + np.random.normal(0, self.angle_noise)

                return np.array([measured_range, measured_angle])

        class IRSensor:
            """红外传感器模型"""

            def __init__(self, node):
                self.node = node  # 保存对父节点的引用
                self.max_range = 150
                self.azimuth_range = np.radians(90)
                self.range_noise = 50
                self.angle_noise = np.radians(1)
                self.detection_prob = 0.85
                self.false_alarm_rate = 0.05

            def detect(self, target_pos):
                """判断是否能检测到目标"""
                dx = target_pos[0] - self.node.pos[0]  # 使用节点位置
                dy = target_pos[1] - self.node.pos[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance > self.max_range:
                    return False

                angle = np.arctan2(dy, dx)
                if abs(angle) > self.azimuth_range / 2:
                    return False

                return np.random.random() < self.detection_prob

            def measure(self, target_pos):
                """生成红外测量值（极坐标）"""
                if np.random.random() < self.false_alarm_rate:
                    false_range = np.random.uniform(0, self.max_range)
                    false_angle = np.random.uniform(
                        -self.azimuth_range / 2, self.azimuth_range / 2
                    )
                    return np.array([false_range, false_angle])

                dx = target_pos[0] - self.node.pos[0]  # 使用节点位置
                dy = target_pos[1] - self.node.pos[1]
                true_range = np.sqrt(dx**2 + dy**2)
                true_angle = np.arctan2(dy, dx)

                measured_range = true_range + np.random.normal(0, self.range_noise)
                measured_angle = true_angle + np.random.normal(0, self.angle_noise)

                return np.array([measured_range, measured_angle])


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器：处理非线性观测模型"""

    def __init__(self, target_id, sensor_type="radar"):
        self.target_id = target_id
        self.sensor_type = sensor_type
        self.dt = 0.1  # 时间步长

        # 状态向量: [x, y, vx, vy, ax, ay]
        self.x = np.zeros(6)  # 初始状态  一维数组全0
        self.P = np.eye(6) * 1000  # 初始协方差矩阵  生成单位矩阵

        # 状态转移矩阵
        self.F = np.array(
            [
                [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],
                [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],
                [0, 0, 1, 0, self.dt, 0],
                [0, 0, 0, 1, 0, self.dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # 过程噪声协方差
        self.Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.001, 0.001])

        # 观测噪声协方差
        if sensor_type == "radar":
            self.R = np.diag([10**2, np.radians(0.5) ** 2])  # 雷达测量噪声
        elif sensor_type == "irst":
            self.R = np.diag([50**2, np.radians(1) ** 2])  # 红外测量噪声
        else:  # 融合
            self.R = np.diag([5**2, 5**2])  # 融合后的测量噪声（假设更低）

    def predict(self, dt):
        """预测步骤"""
        if dt != self.dt:
            self.dt = dt
            self.F = np.array(
                [
                    [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],
                    [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z, sensor_pos):
        """更新步骤（处理非线性观测）"""
        if self.sensor_type in ["radar", "irst"]:
            # 极坐标观测（非线性）
            H = self._calculate_jacobian(self.x, sensor_pos)
            z_pred = self._h(self.x, sensor_pos)  # 预测观测
        else:  # 融合后的线性观测
            H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
            z_pred = H @ self.x

        # 计算残差
        y = z - z_pred

        # 角度残差归一化到[-π, π]
        if self.sensor_type in ["radar", "irst"]:
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

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
        self.predict(self.dt)

        # 计算融合后的测量值（简化的加权平均）
        z_fused = np.zeros(2)#2×1
        R_fused_inv = np.zeros((2, 2))#2×2

        for sensor_type, z in measurements.items():
            if sensor_type == "radar":
                R = np.diag([10**2, np.radians(0.5) ** 2])
            elif sensor_type == "irst":
                R = np.diag([50**2, np.radians(1) ** 2])#R xy=J⋅ Rrθ⋅ JT注意此处是否也需要对R进求偏导得到雅可比J
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
            r = np.sqrt(dx**2 + dy**2)
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
            r_squared = dx**2 + dy**2 + 1e-10
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


class DataLinkTransceiver:
    """数据链收发器：模拟通信过程，包括延迟、丢包和抗干扰"""

    def __init__(self):
        self.freq_hop_seq = np.random.permutation(100)#返回0-99的随机不重复排列
        self.current_ch = 0
        self.packet_loss_rate = 0.1
        self.delay_mean = 0.05
        self.delay_std = 0.02
        self.buffer = {}

    def transmit(self, data, sender_id, receiver_id):
        if np.random.random() < self.packet_loss_rate:
            return None

        delay = np.random.normal(self.delay_mean, self.delay_std)#返回一个符合期望和标准差的正态分布数
        if delay < 0:
            delay = 0

        arrival_time = simulator.time + delay
        key = (arrival_time, sender_id, receiver_id)

        encrypted_data = self._apply_fhss(data)
        self.buffer[key] = encrypted_data

        return arrival_time

    def receive(self, current_time, receiver_id):
        received_data = []
        keys_to_remove = []

        for key in self.buffer:
            arrival_time, sender_id, recv_id = key
            if arrival_time <= current_time and recv_id == receiver_id:
                decrypted_data = self._apply_dehopping(self.buffer[key])
                received_data.append((sender_id, decrypted_data))
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.buffer[key]

        return received_data

    def _apply_fhss(self, data):
        self.current_ch = (self.current_ch + 1) % len(self.freq_hop_seq)
        return data

    def _apply_dehopping(self, data):
        return data


class BattlefieldVisualizer:
    """战场可视化器：显示目标、传感器和跟踪结果"""

    def __init__(self, simulator):
        self.sim = simulator
        self.fig, (self.ax, self.error_ax) = plt.subplots(2, 1, figsize=(10, 12))
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        self.ax.set_xlabel("X坐标 (m)")
        self.ax.set_ylabel("Y坐标 (m)")
        self.ax.set_title("战场目标跟踪模拟")

        # 初始化绘图元素
        self.targets_scat = self.ax.scatter(
            [], [], c="red", marker="o", s=50, label="目标"
        )
        self.nodes_scat = self.ax.scatter(
            [], [], c="black", marker="^", s=100, label="传感器节点"
        )

        # 传感器视场范围
        self.sensor_fovs = []
        for node in self.sim.nodes:
            for sensor_type, sensor in node.sensors.items():
                if sensor_type == "radar":
                    color = "blue"
                    alpha = 0.1
                else:  # IRST
                    color = "green"
                    alpha = 0.05

                fov = plt.Circle(
                    (node.pos[0], node.pos[1]),
                    sensor.max_range,
                    color=color,
                    alpha=alpha,
                    label=f"{sensor_type.upper()} 视场" if not self.sensor_fovs else "",
                )
                self.ax.add_patch(fov)
                self.sensor_fovs.append(fov)

        # 测量点
        (self.radar_meas,) = self.ax.plot(
            [], [], "b.", markersize=4, alpha=0.3, label="雷达测量"
        )
        (self.irst_meas,) = self.ax.plot(
            [], [], "g.", markersize=4, alpha=0.3, label="红外测量"
        )

        # 轨迹线
        self.true_trajs = []
        self.radar_trajs = []
        self.irst_trajs = []
        self.fused_trajs = []

        for target_id in range(self.sim.num_targets):
            (true_line,) = self.ax.plot(
                [],
                [],
                "red",
                linewidth=2,
                alpha=0.8,
                label=f"目标{target_id+1}真实轨迹" if target_id == 0 else "",
            )
            (radar_line,) = self.ax.plot(
                [],
                [],
                "blue",
                linewidth=1.5,
                alpha=0.6,
                label=f"目标{target_id+1}雷达估计" if target_id == 0 else "",
            )
            (irst_line,) = self.ax.plot(
                [],
                [],
                "green",
                linewidth=1.5,
                alpha=0.6,
                label=f"目标{target_id+1}红外估计" if target_id == 0 else "",
            )
            (fused_line,) = self.ax.plot(
                [],
                [],
                "purple",
                linewidth=2,
                alpha=1.0,
                label=f"目标{target_id+1}融合估计" if target_id == 0 else "",
            )

            self.true_trajs.append(true_line)
            self.radar_trajs.append(radar_line)
            self.irst_trajs.append(irst_line)
            self.fused_trajs.append(fused_line)

        # 误差椭圆（协方差可视化）
        self.error_ellipses = []
        for target_id in range(self.sim.num_targets):
            ellipse = Ellipse(
                xy=(0, 0),
                width=2,
                height=2,
                angle=0,
                color="purple",
                alpha=0.3,
                label="融合误差椭圆" if target_id == 0 else "",
            )
            self.ax.add_patch(ellipse)
            self.error_ellipses.append(ellipse)

        # 误差统计图表
        self.error_ax.set_title("目标跟踪误差")
        self.error_ax.set_xlabel("时间步")
        self.error_ax.set_ylabel("误差 (m)")

        self.radar_error_lines = []
        self.irst_error_lines = []
        self.fused_error_lines = []

        for target_id in range(self.sim.num_targets):
            (radar_line,) = self.error_ax.plot(
                [],
                [],
                "blue",
                linewidth=1,
                alpha=0.6,
                label=f"雷达误差 (目标{target_id+1})" if target_id == 0 else "",
            )
            (irst_line,) = self.error_ax.plot(
                [],
                [],
                "green",
                linewidth=1,
                alpha=0.6,
                label=f"红外误差 (目标{target_id+1})" if target_id == 0 else "",
            )
            (fused_line,) = self.error_ax.plot(
                [],
                [],
                "purple",
                linewidth=2,
                alpha=1.0,
                label=f"融合误差 (目标{target_id+1})" if target_id == 0 else "",
            )

            self.radar_error_lines.append(radar_line)
            self.irst_error_lines.append(irst_line)
            self.fused_error_lines.append(fused_line)

        # 历史数据存储
        self.true_pos_history = [[] for _ in range(self.sim.num_targets)]
        self.radar_pos_history = [[] for _ in range(self.sim.num_targets)]
        self.irst_pos_history = [[] for _ in range(self.sim.num_targets)]
        self.fused_pos_history = [[] for _ in range(self.sim.num_targets)]

        self.radar_error_history = [[] for _ in range(self.sim.num_targets)]
        self.irst_error_history = [[] for _ in range(self.sim.num_targets)]
        self.fused_error_history = [[] for _ in range(self.sim.num_targets)]

        # 图例
        self.ax.legend(loc="upper right", fontsize=8)
        self.error_ax.legend(loc="upper right", fontsize=8)

        # 初始化节点滤波器
        for node in self.sim.nodes:
            node.init_filters(self.sim.targets)

    def update(self, frame):
        """动画更新函数"""
        # 更新所有目标
        for target in self.sim.targets:
            target.update()

        # 传感器感知和滤波
        radar_measurements = []
        irst_measurements = []

        for node in self.sim.nodes:
            for target in self.sim.targets:
                measurements = node.sense_and_filter(target, 0.1)

                # 收集测量数据用于可视化
                for sensor_type, z in measurements.items():
                    if sensor_type == "radar":
                        r, theta = z
                        x = node.pos[0] + r * np.cos(theta)
                        y = node.pos[1] + r * np.sin(theta)
                        radar_measurements.append((x, y))
                    elif sensor_type == "irst":
                        r, theta = z
                        x = node.pos[0] + r * np.cos(theta)
                        y = node.pos[1] + r * np.sin(theta)
                        irst_measurements.append((x, y))

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
            radar_filter.predict(0.1)
            irst_filter.predict(0.1)
            fused_filter.predict(0.1)

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
    # 创建战场模拟器
    simulator = BattlefieldSimulator(num_targets=3, num_nodes=5)

    # 创建可视化器
    visualizer = BattlefieldVisualizer(simulator)

    # 定义动画更新函数
    def animate(frame):
        return visualizer.update(frame)

    # 初始化动画
    ani = FuncAnimation(
        visualizer.fig,
        animate,
        frames=np.arange(0, 200),
        interval=50,
        blit=True,
        repeat=False,
    )

    # 显示图例
    visualizer.ax.legend(loc="upper right", fontsize=6)
    visualizer.error_ax.legend(loc="upper right", fontsize=6)

    # 启动动画显示
    plt.tight_layout()
    plt.show()

    # 动画结束后输出精度报告
    visualizer.print_accuracy_report()

