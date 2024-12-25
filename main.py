import numpy as np
import matplotlib.pyplot as plt
import copy
from celluloid import Camera  # 保存动图时用，pip install celluloid
from gym import Env
from gym import spaces  # 为环境数据参数进行刻画
from stable_baselines3 import PPO
from gym.spaces import Dict, Box


class myenv(Env):
    def __init__(self):
        self.current_state = None

        self.d = 3.5

        self.w = 1.8

        self.L = 4.7

        self.P0 = np.array([0, - self.d / 2, 1, 1])

        self.Pg = np.array([99, self.d / 2, 0, 0])

        self.Pobs = np.array([
            [15, 7 / 4, 0, 0],
            [30, - 3 / 2, 0, 0],
            [45, 3 / 2, 0, 0],
            [60, - 3 / 4, 0, 0],
            [80, 3 / 2, 0, 0]])

        self.P = np.vstack((self.Pg, self.Pobs))

        self.Eta_att = 0.1  # 引力的增益系数

        self.Eta_rep_ob = 15  # 斥力的增益系数

        self.Eta_rep_edge = 50  # 道路边界斥力的增益系数

        self.d0 = 20  # 障碍影响的最大距离

        self.num = self.P.shape[0]  # 障碍与目标总计个数

        self.len_step = 0.5  # 步长

        self.n = 1

        self.Num_iter = 200  # 最大循环迭代次数

        self.path = []  # 保存车走过的每个点的坐标

        self.delta = np.zeros((self.num, 2))  # 保存车辆当前位置与障碍物的方向向量，方向指向车辆；以及保存车辆当前位置与目标点的方向向量，方向指向目标点

        self.dists = []  # 保存车辆当前位置与障碍物的距离以及车辆当前位置与目标点的距离

        self.unite_vec = np.zeros((self.num, 2))  # 保存车辆当前位置与障碍物的单位方向向量，方向指向车辆；以及保存车辆当前位置与目标点的单位方向向量，方向指向目标点

        self.F_rep_ob = np.zeros((len(self.Pobs), 2))  # 存储每一个障碍到车辆的斥力,带方向

        self.v = np.linalg.norm(self.P0[2:4])  # 设车辆速度为常值

        ## ***************初始化结束，开始主体循环******************
        self.Pi = self.P0[0:2]  # 当前车辆位置

        self.count = 0

        self.observation_space = Dict({
            "obs1": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "obs2": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "obs3": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "obs4": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "obs5": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "goal": Box(low=-100, high=100, shape=(2,), dtype=np.float32),
            "ego": Box(low=-100, high=100, shape=(2,), dtype=np.float32)
        })

        # 表示纵向速度
        self.action_space = Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def step(self, action):
        self.count += 1
        self.path.append(self.Pi)
        UnitVec_Fsum = np.array([1, action[0]])
        self.Pi = copy.deepcopy(self.Pi + self.len_step * UnitVec_Fsum)
        self.dists = []
        # 计算车辆与障碍物之间的单位方向向量
        for j in range(len(self.Pobs)):
            self.delta[j] = self.Pi[0:2] - self.Pobs[j, 0:2]
            self.dists.append(np.linalg.norm(self.delta[j]))  # 计算欧式距离
            self.unite_vec[j] = self.delta[j] / self.dists[j]  # 计算出x,y方向的单位向量

        # 计算车辆当前位置与目标的单位方向向量
        self.delta[len(self.Pobs)] = self.Pg[0:2] - self.Pi[0:2]
        self.dists.append(np.linalg.norm(self.delta[len(self.Pobs)]))
        self.unite_vec[len(self.Pobs)] = self.delta[len(self.Pobs)] / self.dists[len(self.Pobs)]

        # 引力奖励（目标吸引力）
        F_att_reward = np.exp(-self.dists[len(self.Pobs)] / 10)  # 高斯衰减

        # 障碍物斥力奖励函数设计
        F_rep_ob_reward = 0
        for j in range(len(self.Pobs)):
            if self.dists[j] < self.d0:  # 距离小于最大影响距离
                F_rep_ob_reward += np.exp(-self.dists[j] / 5)  # 高斯衰减的斥力

        F_rep_ob_reward = -F_rep_ob_reward  # 斥力总是负的

        # 边界惩罚
        if self.Pi[1] > self.d or self.Pi[1] < -self.d:
            reward_edg = -50
            done4 = True
        else:
            reward_edg = 0
            done4 = False

        # 碰撞惩罚
        reward_collision = 0
        done1 = False
        for j in range(len(self.Pobs)):
            if self.dists[j] <= 0.5:
                reward_collision = -100
                done1 = True
                break

        # 到达目标奖励
        if self.dists[len(self.Pobs)] < 1:
            reward_goal = 100
            done2 = True
        else:
            reward_goal = 0 # 距离目标点的每一步都略微惩罚
            done2 = False

        # 时间步惩罚
        if self.count > self.Num_iter:
            reward_count = -50
            done3 = True
        else:
            reward_count = -0.1  # 每一步略微惩罚时间
            done3 = False

        done = done1 or done2 or done3 or done4
        reward = (
                F_att_reward + F_rep_ob_reward +
                reward_collision + reward_goal +
                reward_count + reward_edg
        )

        self.current_state = {
            "obs1": self.Pobs[0, 0:2],
            "obs2": self.Pobs[1, 0:2],
            "obs3": self.Pobs[2, 0:2],
            "obs4": self.Pobs[3, 0:2],
            "obs5": self.Pobs[4, 0:2],
            "goal": self.Pg[0:2],
            "ego": self.Pi[0:2]
        }

        return self.current_state, reward, done, {}

    def reset(self):
        self.Pi = self.P0[0:2]
        self.count = 0
        a = self.Pobs[0, 0:2]
        self.current_state = {
            "obs1": self.Pobs[0, 0:2],
            "obs2": self.Pobs[1, 0:2],
            "obs3": self.Pobs[2, 0:2],
            "obs4": self.Pobs[3, 0:2],
            "obs5": self.Pobs[4, 0:2],
            "goal": self.Pg[0:2],
            "ego": self.Pi[0:2]
        }
        return self.current_state


if __name__ == '__main__':
    env = myenv()

    train = False

    if train:

        model = PPO('MultiInputPolicy', env,
                    policy_kwargs=dict(net_arch=[64, 64]),  # 策略网络
                    learning_rate=5e-4,  # 学习率
                    batch_size=512,  # 训练时使用样本批次的大小
                    gamma=0.99,
                    verbose=1,
                    # seed=16,
                    tensorboard_log="test/")

        model.learn(500000, log_interval=1)
        model.save("test/PPOmodel")

    else:

        model = PPO.load("test/PPOmodel", env=env)
        eposides = 1
        for eq in range(eposides):

            obs = env.reset()
            done = False
            rewards = 0
            while not done:
                # action = env.action_space.sample()
                action, _state = model.predict(obs, deterministic=True)
                # action = action.item()
                obs, reward, done, info = env.step(action)
                # env.render()
                rewards += reward
            print(rewards)
            fig = plt.figure(1)
            # plt.ylim(-4, 4)
            plt.axis([-10, 100, -15, 15])
            camera = Camera(fig)
            len_line = 100
            # 画灰色路面图
            GreyZone = np.array([[- 5, - env.d - 0.5], [- 5, env.d + 0.5],
                                 [len_line, env.d + 0.5], [len_line, - env.d - 0.5]])
            for i in range(len(env.path)):
                plt.fill(GreyZone[:, 0], GreyZone[:, 1], 'gray')
                plt.fill(np.array([env.P0[0], env.P0[0], env.P0[0] - env.L, env.P0[0] - env.L]), np.array([- env.d /
                                                                                                           2 - env.w / 2,
                                                                                                           - env.d / 2 + env.w / 2,
                                                                                                           - env.d / 2 + env.w / 2,
                                                                                                           - env.d / 2 - env.w / 2]), 'b')
                # 画分界线
                plt.plot(np.array([- 5, len_line]), np.array([0, 0]), 'w--')

                plt.plot(np.array([- 5, len_line]), np.array([env.d, env.d]), 'w')

                plt.plot(np.array([- 5, len_line]), np.array([- env.d, - env.d]), 'w')

                # 设置坐标轴显示范围
                # plt.axis('equal')
                # plt.gca().set_aspect('equal')
                # 绘制路径
                plt.plot(env.Pobs[:, 0], env.Pobs[:, 1], 'ro')  # 障碍物位置

                plt.plot(env.Pg[0], env.Pg[1], 'gv')  # 目标位置

                plt.plot(env.P0[0], env.P0[1], 'bs')  # 起点位置
                # plt.cla()
                env.path.append(env.Pg[0:2])
                path = np.array(env.path)
                plt.plot(path[0:i, 0], path[0:i, 1], 'k')  # 路径点
                plt.pause(0.001)
