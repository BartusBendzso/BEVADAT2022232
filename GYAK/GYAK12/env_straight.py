import numpy as np
import math
import matplotlib.pyplot as plt


class Straight_env:
    velocity = 15
    dt = 0.1
    lane_width = 10
    action_size = 3
    L = 3
    actions = np.array([0.2, 0, -0.2])
    action = []

    def __init__(self):
        self.x = 0
        self.y = 0
        self.th = 0

    def step(self, action):
        a = self.actions[action]
        self.action = [a, 0]
        self.vehicle_onestep()
        normalized_distance = self.y / self.lane_width
        normalized_relative_yaw_angle = self.th / (math.pi / 2)
        done = False
        reward = self.calculating_reward()
        if abs(normalized_distance) > 1 or self.x > 100 or abs(normalized_relative_yaw_angle) > 1:
            done = True
        return np.array([normalized_distance, -normalized_relative_yaw_angle]), reward, done, {}

    def reset(self):
        self.x = 0
        self.y = np.random.uniform(-5, 5, 1)[0]
        self.th = np.random.uniform(-np.pi/4, np.pi/4, 1)[0]
        plt.cla()
        return np.array([self.y / self.lane_width, self.th / (math.pi / 2)])

    def render(self):
        plt.ylim([-30, 30])
        plt.xlim([0, 100])
        plt.plot([0, 100], [10, 10], c='black')
        plt.plot([0, 100], [-10, -10], c='black')
        plt.scatter(self.x, self.y, c='black')
        plt.pause(0.0001)

    def minmax_rescaling(self, a, b, mi, ma, x):
        return a + (((x - mi) * (b - a)) / (ma - mi))

    def vehicle_onestep(self):
        """
        :param vehiclestate: np.array([x,y,th,v])                                                     
                            x,y - position ([m,m])
                            th  - angle ([rad] zero at x direction,CCW),  th = theta (heading)
                            v   - velocity ([m/s])
        :param action: np.array([steering, acceleration])
                            steering     - angle CCW [rad]
                            acceleration - m/s^2
        :param dt: sample time [s]
        :return:the new vehicle state in same structure as the vehiclestate param
        """
        # The new speed v'=v+dt*a
        self.velocity = max(0, self.velocity + self.dt * self.action[1])
        # The travelled distance s=(v+v')/2*dt
        s = (self.velocity + self.velocity) / 2 * self.dt
        if self.action[0] == 0:  # Not steering
            # unit vector
            dx = math.cos(self.th)
            dy = math.sin(self.th)
            self.x = self.x + dx * s
            self.y = self.y + dy * s
        else:  # Steering
            # Turning Radius R=axlelength/tanh(steering)
            R = self.L / math.tanh(self.action[0])
            # The new theta heading th'=th+s/R
            turn = s / R
            self.th = self.th + turn
            # Normálás
            if math.pi < self.th:
                self.th = self.th - 2 * math.pi
            if -math.pi > self.th:
                self.th = self.th + 2 * math.pi
            # new position
            # transpose distance dist=2*R*sin(|turn/2|)
            dist = abs(2 * R * math.sin(turn / 2))
            # transpose angle ang=th+turn/2
            ang = self.th + turn / 2
            # unit vector
            dx = math.cos(ang)
            dy = math.sin(ang)
            # new position
            self.x = self.x + dx * dist
            self.y = self.y + dy * dist

    def calculating_reward(self):
        if abs(self.y/self.lane_width) > 1:
            return -1
        else:
            return 0


if __name__ == "__main__":
    env = Straight_env()
    env.reset()
    for i in range(100):
        env.step(np.random.randint(0, 3, 1)[0])
        env.render()
