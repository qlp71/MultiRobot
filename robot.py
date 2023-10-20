#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import json
import math
import socket
import time
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import os
import xlrd

get_info_msg = b'\x02\x31\x30\x30\x31\x30\x30\x30\x30\x30\x03'
speed_pub_head = b'\x02\x33\x30\x30\x30\x31'
speed_pub_tail = b'\x30\x30\x30\x30\x03'
nav_info_msg = b'\x02\x36\x30\x30\x30\x33\x30\x30\x30\x30\x03'
# speed_v_max = 1.0
# speed_w_max = 1.0
speed_v_max = 1.2
speed_w_max = 1.2
speed_phi_max = math.pi/2*1
speed_a_max = 5.0
speed_beta_max = 20.0
theta_error_max = math.atan2(0.5, 1)
weight = {"x_m": 1.5, "y_m": 1.5, "theta_m": 0.8,
          "d_k": 1.0, "theta_k": 0.2, "v_x_k": 0.01, "v_y_k": 0.01,
          "v_k": 0.1, "w_k": 0.01, "phi_k": 0.002}
# weight = {"x_m": 1.5, "y_m": 1.5, "theta_m": 0.2,
#           "d_k": 1.0, "theta_k": 0.1,
#           "v_k": 0.1, "w_k": 0.05,
#           "v_x_k": 0.01, "v_y_k": 0.01, "phi_k": 0.01}


class Position:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.xy = np.array([x, y])
        self.xyt = np.array([x, y, theta])

    def trans(self, translate_x, translate_y, rotation_theta):
        return Position(x=self.x * math.cos(rotation_theta) - self.y * math.sin(rotation_theta) + translate_x,
                        y=self.x * math.sin(rotation_theta) + self.y *
                        math.cos(rotation_theta) + translate_y,
                        theta=self.theta + rotation_theta)

    def distance(self, other, option="max", k=0.5):
        if option == "max":
            return max(abs(self.x - other.x), abs(self.y - other.y), k*abs(self.theta - other.theta))
        elif option == "euclidean_Q":
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (k*(self.theta - other.theta)) ** 2)
        elif option == "euclidean_xy":
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def array(self):
        return [self.x, self.y, self.theta]

    def __dict__(self):
        return {"pos_x": self.x, "pos_y": self.y, "pos_theta": self.theta}

    def __str__(self):
        return self.__dict__().__str__()

    def __deepcopy__(self):
        return Position(x=copy.deepcopy(self.x), y=copy.deepcopy(self.y), theta=copy.deepcopy(self.theta))

    def __add__(self, other):
        return Position(x=self.x + other.x, y=self.y + other.y, theta=self.theta + other.theta)

    def __sub__(self, other):
        return Position(x=self.x - other.x, y=self.y - other.y, theta=self.theta - other.theta)


class Speed:
    def __init__(self, v=0.0, w=0.0, v_x=0.0, v_y=0.0, phi=0.0, pub_v=0.0, pub_w=0.0, pub_v_x=0.0, pub_v_y=0.0, pub_phi=0.0):
        self.v = v
        self.w = w
        self.v_x = v_x
        self.v_y = v_y
        self.phi = phi
        self.u = np.array([v, w])
        self.u_a = np.array([v_x, v_y, w])
        self.u_c = np.array([v, phi])
        self.pub_v = pub_v
        self.pub_w = pub_w
        self.pub_v_x = pub_v_x
        self.pub_v_y = pub_v_y
        self.pub_phi = pub_phi
        self.pub_u = np.array([pub_v, pub_w])
        self.pub_u_a = np.array([pub_v_x, pub_v_y, pub_w])
        self.pub_u_c = np.array([pub_v, pub_phi])

    def send_dict(self):
        return {"speed_vx": self.v, "speed_w": self.w}

    def send_str(self):
        return self.send_dict().__str__()

    def array(self, flag="real", type=0):
        if flag == "real":
            if type == 0:
                return [self.v, self.w]
            elif type == 1:
                return [self.v_x, self.v_y, self.w]
            elif type == 2:
                return [self.v, self.phi]
        elif flag == "pub":
            if type == 0:
                return [self.pub_v, self.pub_w]
            elif type == 1:
                return [self.pub_v_x, self.pub_v_y, self.pub_w]
            elif type == 2:
                return [self.pub_v, self.pub_phi]

    def __dict__(self):
        return {"v": self.v, "w": self.w, "pub_v": self.pub_v, "pub_w": self.pub_w}

    def __str__(self):
        return self.__dict__().__str__()

    def __deepcopy__(self):
        return Speed(copy.deepcopy(self.v), copy.deepcopy(self.w))


class Robot:
    def __init__(self, pos=Position(0, 0, 0), speed=Speed(0, 0, 0, 0), ip_addr="localhost", ports=None, **kwargs):
        self.ip_addr = ip_addr
        self.name = ip_addr[-3:]
        if not ports:
            self.ports = {"status": {"name": 18204, "client": socket.socket(), "connect": False},
                          "publish": {"name": 18207, "client": socket.socket(), "connect": False},
                          "nav": {"name": 18206, "client": socket.socket(), "connect": False}}
        else:
            pass
        self.pos = pos
        self.speed = speed
        self.status = "init"
        self.present_time = time.time()
        self.history = {"time": [], "pos": [], "speed": [], "speed_pub": []}
        self.history_switch = True
        self.color = kwargs.get("color", "b")
        self.emc_status = 0
        self.T = -1
        self.filter = [False, 5*self.T]
        # 0: 双轮差速
        # 1: 全向移动
        # 2: carlike
        self.type = kwargs.get("type", 0)
        # carlike 的前后轮间距（米）
        self.L = 1.0

    def enable_filter(self, a=7.0):
        self.filter[0] = True
        self.filter[1] = a*self.T

    def connect(self, port_name):
        print("connecting...")
        self.ports[port_name]["client"].connect(
            (self.ip_addr, self.ports[port_name]["name"]))
        print("connected")
        self.ports[port_name]["connect"] = True
        return True

    def disconnect(self, port_name):
        self.ports[port_name]["client"].close()
        self.ports[port_name]["connect"] = False
        return True

    def close(self):
        self.disconnect("status")
        self.disconnect("publish")
        return True

    def read_receive_msg(self, receive_msg):
        receive_msg_dict = json.loads(receive_msg)
        theta_temp = receive_msg_dict["data"]["position"]["pos_angle"]
        self.emc_status = receive_msg_dict["data"]["emec"]["emc_status"]
        # front_sensor = receive_msg_dict["data"]["isFrontSensor"]
        # rear_sensor = receive_msg_dict["data"]["isRearSensor"]
        # print(em_stop)
        if self.status == "init":
            pass
        else:
            self.status = "reading message"
            flag_plus = 0
            while np.abs(theta_temp - self.pos.theta) >= speed_w_max * 0.5:
                if theta_temp - self.pos.theta > math.pi:
                    if flag_plus <= 0:
                        theta_temp -= 2 * math.pi
                        flag_plus = -1
                    else:
                        print("error theta")
                        return False
                elif theta_temp - self.pos.theta < math.pi:
                    if flag_plus >= 0:
                        theta_temp += 2 * math.pi
                        flag_plus = 1
                    else:
                        print("error theta")
                        return False
        self.pos.__init__(receive_msg_dict["data"]["position"]["pos_x"],
                          receive_msg_dict["data"]["position"]["pos_y"],
                          theta_temp)
        self.speed.__init__(v=receive_msg_dict["data"]["speed"]["speed_vx"],
                            w=receive_msg_dict["data"]["speed"]["speed_w"],
                            pub_v=self.speed.pub_v, pub_w=self.speed.pub_w)
        # print(receive_msg_dict)
        # if self.history_switch:
        #     self.history["pos"].append(copy.deepcopy(self.pos.array()))
        #     self.history["speed"].append(
        #         copy.deepcopy(self.speed.array("real")))
        #     self.history["speed_pub"].append(copy.deepcopy(self.speed.array("pub")))
        #     self.history["time"].append(copy.deepcopy(self.present_time))
        return True

    def get_position(self):
        client = self.ports["status"]["client"]
        if not self.ports["status"]["connect"]:
            self.connect(port_name="status")
        tic = time.time()
        client.send(get_info_msg)
        receive_msg = client.recv(2048)
        receive_msg = receive_msg[6:-6]
        if self.read_receive_msg(receive_msg):
            self.present_time = time.time()
            toc = time.time()
            # return receive_msg, toc - tic
            return toc - tic
        else:
            return False

    def publish_speed(self):
        client = self.ports["publish"]["client"]
        if not self.ports["publish"]["connect"]:
            self.connect("publish")
        if self.filter[0]:
            vx = self.filter[1]*self.speed.pub_v + \
                (1-self.filter[1])*self.speed.v
            vt = self.filter[1]*self.speed.pub_w + \
                (1-self.filter[1])*self.speed.w
            self.speed.pub_v = vx
            self.speed.pub_w = vt
            self.speed.pub_u = np.array([vx, vt])
        else:
            vx = self.speed.pub_v
            vt = self.speed.pub_w
        speed_pub = {"vx": vx, "vy": 0.0, "vtheta": vt}
        tic = time.time()
        # print(speed_pub.__str__().replace("'", "\""))
        self.status = "publishing speed"
        client.send(speed_pub_head + speed_pub.__str__().replace("'",
                    "\"").encode() + speed_pub_tail)
        toc = time.time()
        receive_msg = client.recv(1024)
        if self.history_switch:
            self.history["pos"].append(copy.deepcopy(self.pos.array()))
            self.history["speed"].append(
                copy.deepcopy(self.speed.array("real")))
            self.history["speed_pub"].append(
                copy.deepcopy(self.speed.array("pub")))
            self.history["time"].append(copy.deepcopy(self.present_time))
        return receive_msg[6:-6], toc - tic

    def simu_predict(self, delta_t=0.1):
        if self.type == 0:
            if self.filter[0]:
                self.speed.pub_v = self.filter[1] * \
                    self.speed.pub_v+(1-self.filter[1])*self.speed.v
                self.speed.pub_w = self.filter[1] * \
                    self.speed.pub_w+(1-self.filter[1])*self.speed.w
            self.speed.v = self.speed.pub_v
            self.speed.w = self.speed.pub_w
            # self.pos.x += self.speed.pub_v * math.cos(self.pos.theta) * delta_t
            # self.pos.y += self.speed.pub_v * math.sin(self.pos.theta) * delta_t
            # self.pos.theta += self.speed.pub_w * delta_t
            if np.abs(self.speed.pub_w) < 1e-8:
                self.pos.x += self.speed.pub_v * \
                    math.cos(self.pos.theta) * delta_t
                self.pos.y += self.speed.pub_v * \
                    math.sin(self.pos.theta) * delta_t
                self.pos.theta += self.speed.pub_w * delta_t
            else:
                self.pos.x += self.speed.pub_v / self.speed.pub_w * (
                    math.sin(self.pos.theta + self.speed.pub_w * delta_t) - math.sin(self.pos.theta))
                self.pos.y += self.speed.pub_v / self.speed.pub_w * (
                    -math.cos(self.pos.theta + self.speed.pub_w * delta_t) + math.cos(self.pos.theta))
                self.pos.theta += self.speed.pub_w * delta_t
        elif self.type == 1:
            if self.filter[0]:
                self.speed.pub_v_x = self.filter[1] * \
                    self.speed.pub_v_x+(1-self.filter[1])*self.speed.v_x
                self.speed.pub_v_y = self.filter[1] * \
                    self.speed.pub_v_y+(1-self.filter[1])*self.speed.v_y
                self.speed.pub_w = self.filter[1] * \
                    self.speed.pub_w+(1-self.filter[1])*self.speed.w
            self.speed.v_x = self.speed.pub_v_x
            self.speed.v_y = self.speed.pub_v_y
            self.speed.w = self.speed.pub_w
            # self.pos.x += self.speed.pub_v * math.cos(self.pos.theta) * delta_t
            # self.pos.y += self.speed.pub_v * math.sin(self.pos.theta) * delta_t
            # self.pos.theta += self.speed.pub_w * delta_t
            self.pos.x += (self.speed.pub_v_x*math.cos(self.pos.theta) -
                           self.speed.pub_v_y*math.sin(self.pos.theta))*delta_t
            self.pos.y += (self.speed.pub_v_x*math.sin(self.pos.theta) +
                           self.speed.pub_v_y*math.cos(self.pos.theta))*delta_t
            self.pos.theta += self.speed.pub_w * delta_t
        elif self.type == 2:
            if self.filter[0]:
                self.speed.pub_v = self.filter[1] * \
                    self.speed.pub_v+(1-self.filter[1])*self.speed.v
                self.speed.pub_phi = self.filter[1] * \
                    self.speed.pub_phi+(1-self.filter[1])*self.speed.phi
            self.speed.v = self.speed.pub_v
            self.speed.phi = self.speed.pub_phi
            w = self.speed.pub_v / self.L * math.tan(self.speed.pub_phi)
            if np.abs(w) < 1e-8:
                self.pos.x += self.speed.pub_v * \
                    math.cos(self.pos.theta) * delta_t
                self.pos.y += self.speed.pub_v * \
                    math.sin(self.pos.theta) * delta_t
                self.pos.theta += w * delta_t
            else:
                self.pos.x += self.speed.pub_v / w * \
                    (math.sin(self.pos.theta + w * delta_t) - math.sin(self.pos.theta))
                self.pos.y += self.speed.pub_v / w * \
                    (-math.cos(self.pos.theta + w * delta_t) +
                     math.cos(self.pos.theta))
                self.pos.theta += w * delta_t
        if self.history_switch:
            self.history["pos"].append(copy.deepcopy(self.pos.array()))
            self.history["speed"].append(
                copy.deepcopy(self.speed.array("real", type=self.type)))
            self.history["speed_pub"].append(
                copy.deepcopy(self.speed.array("pub", type=self.type)))
            self.history["time"].append(copy.deepcopy(self.present_time))
            self.present_time = time.time()

    def __str__(self):
        temp = {"ip_address": self.ip_addr, "potrs": self.ports}
        temp.update({"position": self.pos.__dict__()})
        temp.update({"speed": self.speed.__dict__()})
        return temp.__str__()

    def __deepcopy__(self):
        # if agv_copy is None:
        # agv_copy = AGV(ip_address=copy.deepcopy(self.ip_address),
        #                ports=copy.deepcopy(self.ports),
        #                pos=self.pos.__deepcopy__(),
        #                speed=self.speed.__deepcopy__())
        robot_copy = Robot(ip_addr=copy.deepcopy(self.ip_addr),
                           ports=None,
                           pos=self.pos.__deepcopy__(),
                           speed=self.speed.__deepcopy__())
        return robot_copy

    def record_history(self, file_name):
        # record the history status of the robot
        # json and excel
        json_file = file_name + ".json"
        excel_file = file_name + ".xlsx"
        if not self.history_switch:
            return False
        with open(json_file, "w") as f:
            f.write(json.dumps(self.history))
        data = self.history
        start_time = data["time"][0]
        df = pd.DataFrame(data)
        df['x'] = df['pos'].apply(lambda x: x[0])
        df['y'] = df['pos'].apply(lambda x: x[1])
        df['theta'] = df['pos'].apply(lambda x: x[2])
        if self.type == 0:
            df['v'] = df['speed'].apply(lambda x: x[0])
            df['w'] = df['speed'].apply(lambda x: x[1])
            df['pub_v'] = df['speed_pub'].apply(lambda x: x[0])
            df['pub_w'] = df['speed_pub'].apply(lambda x: x[1])
        elif self.type == 1:
            df['v_x'] = df['speed'].apply(lambda x: x[0])
            df['v_y'] = df['speed'].apply(lambda x: x[1])
            df['w'] = df['speed'].apply(lambda x: x[2])
            df['pub_v_x'] = df['speed_pub'].apply(lambda x: x[0])
            df['pub_v_y'] = df['speed_pub'].apply(lambda x: x[1])
            df['pub_w'] = df['speed_pub'].apply(lambda x: x[2])
        elif self.type == 2:
            df['v'] = df['speed'].apply(lambda x: x[0])
            df['phi'] = df['speed'].apply(lambda x: x[1])
            df['pub_v'] = df['speed_pub'].apply(lambda x: x[0])
            df['pub_phi'] = df['speed_pub'].apply(lambda x: x[1])
        df['t'] = df['time'].apply(lambda x: x-start_time)
        df = df.drop(['time', 'pos', 'speed', 'speed_pub'], axis=1)
        df.to_excel(excel_file)
        return True

    # plot the trajectory of the robot
    def plot_trajectory(self, ax=None, color=None, linewidth=1, linestyle="-"):
        if not color:
            color = self.color
        if not ax:
            ax = plt.gca()
        if self.history_switch:
            x = [pos[0] for pos in self.history["pos"]]
            y = [pos[1] for pos in self.history["pos"]]
            ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle)
        return ax
    # plot the speed of the robot

    def plot_control_input(self, ax=None, color='b', linewidth=1, linestyle='-'):
        if not ax:
            ax = plt.gca()
        if self.history_switch:
            # t = [time[0] for time in self.history["time"]]
            t = self.history["time"]
            v = self.history["speed"]
            ax.plot(t, v, color=color, linewidth=linewidth, linestyle=linestyle)
        return ax

    def nav_info_get(self):
        client = self.ports["nav"]["client"]
        if not self.ports["nav"]["connect"]:
            self.connect(port_name="nav")
        client.send(nav_info_msg)
        receive_msg = client.recv(2048)
        receive_msg = receive_msg[6:-6]
        return receive_msg

    def mission_start(self, str: id):
        pass


def distance_position(pos1=None, pos2=None):
    # calculate the distance between two positions
    if not pos1:
        pos1 = Position(0, 0, 0)
    if not pos2:
        pos2 = Position(0, 0, 0)
    return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)


def angle_position(pos1=None, pos2=None):
    # calculate the angle between two positions
    if not pos1:
        pos1 = Position(0, 0, 0)
    if not pos2:
        pos2 = Position(0, 0, 0)
    return math.atan2(pos1.y - pos2.y, pos1.x - pos2.x)


def distance_robots(robot1=None, robot2=None):
    # calculate the distance between two robots
    if not robot1:
        robot1 = Robot(pos=Position(0, 0, 0))
    if not robot2:
        robot2 = Robot(pos=Position(0, 0, 0))
    return distance_position(robot1.pos, robot2.pos)


def angle_robots(robot1=None, robot2=None):
    # calculate the angle between two robots
    if not robot1:
        robot1 = Robot(pos=Position(0, 0, 0))
    if not robot2:
        robot2 = Robot(pos=Position(0, 0, 0))
    return angle_position(robot1.pos, robot2.pos)


def angle_vector(vector1=None, vector2=None):
    # calculate the angle between two vectors
    if not vector1:
        vector1 = [0, 0]
    if not vector2:
        vector2 = [0, 0]
    if abs(vector1[0] * vector2[1] - vector1[1] * vector2[0]) < 1e-10:
        return 0
    else:
        return math.atan2(vector1[0] * vector2[1] - vector1[1] * vector2[0],
                          vector1[0] * vector2[0] + vector1[1] * vector2[1])


def angle_robots_edge(edge0=None, edge1=None):
    # calculate the angle between initial edge and the present edge
    if not edge0:
        edge0 = [Position(), Position()]
    if not edge1:
        edge1 = [Position(), Position()]
    return angle_vector(vector1=[edge0[0].x - edge0[1].x, edge0[0].y - edge0[1].y],
                        vector2=[edge1[0].x - edge1[1].x, edge1[0].y - edge1[1].y])


def theta_mod(theta1, theta0, error_max=speed_w_max * 0.3):
    # transform the angle to the range of near present angle
    # 把theta1转换到theta0附近
    flag_plus = 0
    flag_error = 0
    while np.abs(theta1 - theta0) >= error_max:
        if theta1 - theta0 > math.pi:
            if flag_plus <= 0:
                theta1 -= 2 * math.pi
                flag_plus = -1
            else:
                flag_error = 1
                break
                # return False
        elif theta1 - theta0 < math.pi:
            if flag_plus >= 0:
                theta1 += 2 * math.pi
                flag_plus = 1
            else:
                flag_error = 1
                break
                # return False
    return theta1, flag_error


def trans_to_position(arg=None):
    # transform the argument to the position
    res = Position(0, 0, 0)
    if type(arg) == Robot:
        res = arg.pos.__deepcopy__()
    elif type(arg) == Position:
        res = arg.__deepcopy__()
    return res


def solve_qp(H, f, Aneq=None, bneq=None, Aeq=None, beq=None, lb=None, ub=None):
    # solve the quadratic programming problem
    H = cvxopt.matrix(H)
    f = cvxopt.matrix(f)
    H = 1 / 2 * (H + H.T)
    [n, _] = H.size
    # equality constraint
    if Aeq is not None:
        Aeq = cvxopt.matrix(np.matrix(Aeq).T)
        beq = cvxopt.matrix(np.matrix(beq).T)
    Albub = None
    lbub = None
    # inequality constraint
    if lb is not None and ub is not None:
        Albub = np.vstack((-np.eye(n), np.eye(n)))
        lbub = np.hstack((-lb, ub)).T
    if Aneq is not None or Albub is not None:
        if Aneq is not None:
            if Albub is not None:
                Aneq = cvxopt.matrix(np.vstack((Aneq, Albub)))
                bneq = cvxopt.matrix(np.vstack((bneq, lbub)))
            else:
                Aneq = cvxopt.matrix(Aneq)
                bneq = cvxopt.matrix(bneq)
        else:
            Aneq = cvxopt.matrix(Albub)
            bneq = cvxopt.matrix(lbub)
    # kwargs = {"P": H, "q": f, "G": A, "h": b, "A": Aeq, "b": beq, "options": {"show_progress": False}}
    kwargs = {"options": {"show_progress": False}}
    # sol = cvxopt.solvers.qp(**kwargs)
    sol = cvxopt.solvers.qp(P=H, q=f, G=Aneq, h=bneq, A=Aeq, b=beq, **kwargs)
    # sol = cvxopt.solvers.qp(P=H, q=f, G=A, h=b.T, A=Aeq, b=beq)
    return sol['x']


def cal_kappa(x, y):
    # calculate the curvature of the path
    # x,y are the coordinates of the path, the length of x and y are 4
    x_dot = np.diff(x)
    y_dot = np.diff(y)
    x_dot2 = np.diff(x_dot)
    y_dot2 = np.diff(y_dot)
    x_dot = np.mean(x_dot)
    y_dot = np.mean(y_dot)
    kappa = (x_dot2 * y_dot - y_dot2 * x_dot) / \
        (x_dot ** 2 + y_dot ** 2) ** 1.5
    return kappa[0]


class RobotsSystem:
    # the class of robots system
    def __init__(self, robots=None, Neg=-1):
        if not robots:
            robots = [Robot(), Robot()]
        self.type = robots[0].type
        self.Nrb = len(robots)
        self.status = "init"
        self.emec = 0
        if Neg >= self.Nrb * 2 - 3:
            self.Neg = Neg
        else:
            self.Neg = self.Nrb * 2 - 3
        self.Nst = 3 + self.Nrb + self.Neg
        self.robots = robots
        self.nodes_head = [[] for _ in range(self.Nrb)]
        self.nodes_tail = [[] for _ in range(self.Nrb)]
        self.nodes_ht = [[] for _ in range(self.Nrb)]
        self.edges = []
        self.mpcOK = True
        self.init_positions = []
        self.sum_init_x = 0.0
        self.sum_init_y = 0.0
        self.x_m = 0.0
        self.x_mr = 0.0
        self.y_m = 0.0
        self.y_mr = 0.0
        self.v_m = 0.0
        self.v_mr = 0.0
        self.v_m_theta = 0.0
        self.v_mr_theta = 0.0
        self.kappa_r = 0.0
        self.theta_m = 0.0
        self.sin_theta_m = 0.0
        self.cos_theta_m = 0.0
        self.theta_mr = 0.0
        self.theta_mr_dot1 = 0.0
        self.w_mr = 0.0
        self.pos = Position(x=self.x_m, y=self.y_m, theta=self.v_m_theta)
        self.pos_r = Position(x=self.x_mr, y=self.y_mr, theta=self.v_mr_theta)
        # self.v_mr = 0.0
        self.theta_i = np.zeros((1, self.Nrb))
        self.theta_ir = np.zeros((1, self.Nrb))
        self.sin_theta_i = np.zeros((1, self.Nrb))
        self.sin_theta_ir = np.zeros((1, self.Nrb))
        self.cos_theta_i = np.zeros((1, self.Nrb))
        self.cos_theta_ir = np.zeros((1, self.Nrb))
        if self.type == 0:
            self.v_i = np.zeros((1, self.Nrb))
            self.v_ir = np.zeros((1, self.Nrb))
            self.w_i = np.zeros((1, self.Nrb))
            self.w_ir = np.zeros((1, self.Nrb))
            self.u = np.zeros((1, self.Nrb * 2))
            self.u_r = np.zeros((1, self.Nrb * 2))
        elif self.type == 1:
            self.vx_i = np.zeros((1, self.Nrb))
            self.vx_ir = np.zeros((1, self.Nrb))
            self.vy_i = np.zeros((1, self.Nrb))
            self.vy_ir = np.zeros((1, self.Nrb))
            self.w_i = np.zeros((1, self.Nrb))
            self.w_ir = np.zeros((1, self.Nrb))
            self.u = np.zeros((1, self.Nrb * 3))
            self.u_r = np.zeros((1, self.Nrb * 3))
        elif self.type == 2:
            self.v_i = np.zeros((1, self.Nrb))
            self.v_ir = np.zeros((1, self.Nrb))
            self.phi_i = np.zeros((1, self.Nrb))
            self.phi_ir = np.zeros((1, self.Nrb))
            self.u = np.zeros((1, self.Nrb * 2))
            self.u_r = np.zeros((1, self.Nrb * 2))
        self.d_k = np.zeros((1, self.Neg))
        self.d_kr = np.zeros((1, self.Neg))
        self.phi_k = np.zeros((1, self.Neg))
        self.phi_kr = np.zeros((1, self.Neg))
        self.theta_ir_theta_mr_phi_kr = np.zeros((self.Nrb, self.Neg))
        self.cos_theta_ir_theta_mr_phi_kr = np.zeros((self.Nrb, self.Neg))
        self.sin_theta_ir_theta_mr_phi_kr = np.zeros((self.Nrb, self.Neg))
        self.theta_m_phi_k = np.zeros((1, self.Neg))
        self.cos_theta_m_phi_kr = np.zeros((1, self.Neg))
        self.sin_theta_m_phi_kr = np.zeros((1, self.Neg))
        self.sum_phi_kr = 0.0
        self.X_real = np.hstack((
            np.array([[self.x_m, self.y_m, self.theta_m]]
                     ), self.d_k, self.theta_i
        ))
        self.X_ref = np.hstack((
            np.array([[self.x_mr, self.y_mr, self.theta_mr]]
                     ), self.d_kr, self.theta_ir
        ))
        self.history = {"time": [], "X_real": [],
                        "X_ref": [], "u": [], "u_r": []}
        self.history_switch = True
        self.last_tr_features = []
        self.T = -1
        self.missions = {}

    def states_update(self):
        self.x_m = 0.0
        self.y_m = 0.0
        self.emec = 0
        for rb in self.robots:
            if rb.emc_status != 0:
                self.emec = 1
                break
        for i in range(self.Nrb):
            self.x_m += self.robots[i].pos.x
            self.y_m += self.robots[i].pos.y
            self.theta_i[0][i] = self.robots[i].pos.theta
            self.sin_theta_i[0][i] = math.sin(self.theta_i[0][i])
            self.cos_theta_i[0][i] = math.cos(self.theta_i[0][i])
            if self.type == 0:
                self.v_i[0][i] = self.robots[i].speed.v
                self.u[0][i] = self.v_i[0][i]
                self.w_i[0][i] = self.robots[i].speed.w
                self.u[0][i + self.Nrb] = self.w_i[0][i]
            elif self.type == 1:
                self.vx_i[0][i] = self.robots[i].speed.v_x
                self.vy_i[0][i] = self.robots[i].speed.v_y
                self.u[0][i] = self.vx_i[0][i]
                self.u[0][i+self.Nrb] = self.vy_i[0][i]
                self.w_i[0][i] = self.robots[i].speed.w
                self.u[0][i + self.Nrb*2] = self.w_i[0][i]
            elif self.type == 2:
                self.v_i[0][i] = self.robots[i].speed.v
                self.u[0][i] = self.v_i[0][i]
                self.phi_i[0][i] = self.robots[i].speed.phi
                self.u[0][i + self.Nrb] = self.phi_i[0][i]

        self.x_m /= self.Nrb
        self.y_m /= self.Nrb
        if self.status == "init":
            theta_m0 = angle_robots_edge(
                edge0=[self.init_positions[self.edges[0][0]],
                       self.init_positions[self.edges[0][1]]],
                edge1=[self.robots[self.edges[0][0]].pos, self.robots[self.edges[0][1]].pos])
        elif self.status == "working":
            theta_m0 = self.theta_mr
        for k in range(self.Neg):
            self.d_k[0][k] = distance_robots(
                self.robots[self.edges[k][0]], self.robots[self.edges[k][1]])
            theta_m_phi_temp = angle_robots(
                self.robots[self.edges[k][0]], self.robots[self.edges[k][1]])
            theta_m_temp = angle_robots_edge(
                edge0=[self.init_positions[self.edges[k][0]],
                       self.init_positions[self.edges[k][1]]],
                edge1=[self.robots[self.edges[k][0]].pos, self.robots[self.edges[k][1]].pos])
            if self.status != "init":
                theta_m_phi_temp, error_flag = theta_mod(
                    theta_m_phi_temp, self.phi_k[0][k], theta_error_max)
            # else:
            #     pass
            # TMD, 下一行缩进错了，一直没有执行过
            theta_m_temp, error_flag = theta_mod(
                theta_m_temp, theta_m0, theta_error_max)
            self.phi_k[0][k] = theta_m_temp + self.phi_kr[0][k]
            self.theta_m_phi_k[0][k] = theta_m_phi_temp
        self.theta_m = (np.sum(self.phi_k) - self.sum_phi_kr) / self.Neg
        self.x_m += (-math.cos(self.theta_m)*self.sum_init_x +
                     math.sin(self.theta_m)*self.sum_init_y)/self.Nrb
        self.y_m += (-math.sin(self.theta_m)*self.sum_init_x -
                     math.cos(self.theta_m)*self.sum_init_y)/self.Nrb
        self.sin_theta_m = math.sin(self.theta_m)
        self.cos_theta_m = math.cos(self.theta_m)
        self.X_real = np.hstack((
            np.array([[self.x_m, self.y_m, self.theta_m]]
                     ), self.d_k, self.theta_i
        ))
        v_my = 0.0
        v_mx = 0.0
        for i in range(self.Nrb):
            v_my += self.robots[i].speed.v*math.sin(self.theta_i[0][i])
            v_mx += self.robots[i].speed.v*math.cos(self.theta_i[0][i])
        self.v_m_theta = math.atan2(v_my, v_mx)
        self.pos = Position(x=self.x_m, y=self.y_m, theta=self.v_m_theta)
        if self.history_switch & (self.status == "working"):
            # record history from array type to list type
            self.history["time"].append(time.time())
            self.history["X_real"].append(self.X_real[0].tolist())
            self.history["X_ref"].append(self.X_ref[0].tolist())
            self.history["u"].append(self.u[0].tolist())
            self.history["u_r"].append(self.u_r[0].tolist())
        return True

    def record_history(self, file_name):
        # record history into a json file and a excel file
        json_file = file_name+".json"
        excel_file = file_name+".xlsx"
        with open(json_file, 'w') as f:
            json.dump(self.history, f)
        df = pd.DataFrame(self.history)
        # descompose X_real into x_m,y_m,theta_m,d_k,theta_i
        df["x_m"] = df["X_real"].apply(lambda x: x[0])
        df["y_m"] = df["X_real"].apply(lambda x: x[1])
        df["theta_m"] = df["X_real"].apply(lambda x: x[2])
        for k in range(self.Neg):
            df["d_"+str(k)] = df["X_real"].apply(lambda x: x[3+k])
        for i in range(self.Nrb):
            df["theta_i" +
                str(i)] = df["X_real"].apply(lambda x: x[3+self.Neg+i])
        df.drop(columns=["X_real"], inplace=True)
        # descompose X_ref into x_rm,y_rm,theta_rm,d_rk,theta_ri
        df["x_rm"] = df["X_ref"].apply(lambda x: x[0])
        df["y_rm"] = df["X_ref"].apply(lambda x: x[1])
        df["theta_rm"] = df["X_ref"].apply(lambda x: x[2])
        for k in range(self.Neg):
            df["d_r"+str(k)] = df["X_ref"].apply(lambda x: x[3+k])
        for i in range(self.Nrb):
            df["theta_ri" +
                str(i)] = df["X_ref"].apply(lambda x: x[3+self.Neg+i])
        df.drop(columns=["X_ref"], inplace=True)
        if self.type == 0:
            # descompose u into v_i,w_i
            for i in range(self.Nrb):
                df["v_"+str(i)] = df["u"].apply(lambda x: x[i])
                df["w_"+str(i)] = df["u"].apply(lambda x: x[i+self.Nrb])
            df.drop(columns=["u"], inplace=True)
        # descompose u_r into v_r,w_r
            for i in range(self.Nrb):
                df["v_r"+str(i)] = df["u_r"].apply(lambda x: x[i])
                df["w_r"+str(i)] = df["u_r"].apply(lambda x: x[i+self.Nrb])
        elif self.type == 1:
            # descompose u into v_i,w_i
            for i in range(self.Nrb):
                df["vx_"+str(i)] = df["u"].apply(lambda x: x[i])
                df["vy_"+str(i)] = df["u"].apply(lambda x: x[i+self.Nrb])
                df["w_"+str(i)] = df["u"].apply(lambda x: x[i+self.Nrb*2])
            df.drop(columns=["u"], inplace=True)
        # descompose u_r into v_r,w_r
            for i in range(self.Nrb):
                df["vx_r"+str(i)] = df["u_r"].apply(lambda x: x[i])
                df["vy_r"+str(i)] = df["u_r"].apply(lambda x: x[i+self.Nrb])
                df["w_r"+str(i)] = df["u_r"].apply(lambda x: x[i+self.Nrb])
        elif self.type == 2:
            # descompose u into v_i,w_i
            for i in range(self.Nrb):
                df["v_"+str(i)] = df["u"].apply(lambda x: x[i])
                df["phi_"+str(i)] = df["u"].apply(lambda x: x[i+self.Nrb])
            df.drop(columns=["u"], inplace=True)
        # descompose u_r into v_r,w_r
            for i in range(self.Nrb):
                df["v_r"+str(i)] = df["u_r"].apply(lambda x: x[i])
                df["phi_r"+str(i)] = df["u_r"].apply(lambda x: x[i+self.Nrb])
        # time substract the first time to make the time start from 0
        df["time"] = df["time"].apply(lambda x: x-self.history["time"][0])
        df.drop(columns=["u_r"], inplace=True)
        df.to_excel(excel_file, index=False)
        return True

    # plot the trajectorys of the robots
    def plot_trajectorys(self, **kwargs):
        plt.figure()
        plt.title("trajectorys")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid()
        # plot midpoints trajectory
        plt.plot([x[0] for x in self.history["X_real"]],
                 [x[1] for x in self.history["X_real"]], "-")
        plt.plot([x[0] for x in self.history["X_ref"]],
                 [x[1] for x in self.history["X_ref"]], "-")
        # plot robots trajectory
        for rb in self.robots:
            plt.plot([x[0] for x in rb.history["pos"]],
                     [x[1] for x in rb.history["pos"]], "-", color=rb.color)
        is_show = kwargs.get("is_show", True)
        if is_show:
            plt.show()

    # plot the positions of the robots, and show the edges
    def plot_robots_edges(self, gap=10):
        plt.figure()
        plt.title("positions")
        plt.xlabel("x")
        plt.ylabel("y")
        # set the axis equal
        plt.axis("equal")
        plt.grid()
        # plot midpoints
        plt.plot([x[0] for x in self.history["X_real"]],
                 [x[1] for x in self.history["X_real"]], "-")
        # plt.plot([x[0] for x in self.history["X_ref"]],
        #          [x[1] for x in self.history["X_ref"]], "o")
        # plot robots, but only plot one point every gap time steps
        for i in range(np.floor(len(self.history["time"])/gap).astype(int)):
            for rb in self.robots:
                plt.plot(rb.history["pos"][i*gap][0],
                         rb.history["pos"][i*gap][1], "o-", color=rb.color)
        # for rb in self.robots:
        #     plt.plot([x[0] for x in rb.history["pos"]],
        #              [x[1] for x in rb.history["pos"]], "o-")
        # plot edges all the time, but only plot one point every gap time steps
        for i in range((np.floor(len(self.history["time"])/gap)).astype(int)):
            for k in range(self.Neg):
                plt.plot([self.robots[self.edges[k][0]].history["pos"][i*gap][0],
                          self.robots[self.edges[k][1]].history["pos"][i*gap][0]],
                         [self.robots[self.edges[k][0]].history["pos"][i*gap][1],
                         self.robots[self.edges[k][1]].history["pos"][i*gap][1]],
                         "k-", linewidth=0.2)
        plt.show()

    # plot the inputs of the robots
    def plot_control_input(self):
        plt.figure()
        plt.title("control inputs")
        plt.xlabel("time")
        plt.ylabel("input")
        for i in range(self.Nrb):
            plt.plot(self.history["time"], [x[i]
                     for x in self.history["u"]], "-")
            plt.plot(self.history["time"], [x[i+self.Nrb]
                     for x in self.history["u_r"]], "-")
        plt.show()

    def formation_setting(self, init_positions, edges):
        self.sum_phi_kr = 0.0
        if len(edges) == self.Neg:
            if len(init_positions) == self.Nrb:
                self.edges = edges.copy()
                for k in range(self.Neg):
                    self.d_kr[0][k] = distance_position(
                        init_positions[edges[k][0]], init_positions[edges[k][1]])
                    self.phi_kr[0][k] = angle_position(
                        init_positions[edges[k][0]], init_positions[edges[k][1]])
                    self.sum_phi_kr += self.phi_kr[0][k]
                    self.nodes_head[edges[k][0]].append(k)
                    self.nodes_tail[edges[k][1]].append(k)
                    self.nodes_ht[edges[k][0]].append(k)
                    self.nodes_ht[edges[k][1]].append(k)
                self.init_positions = init_positions.copy()
                for i in range(self.Nrb):
                    self.sum_init_x += init_positions[i].x
                    self.sum_init_y += init_positions[i].y
            else:
                print("the number of init_positions is wrong")
                return False
        else:
            print("the number of edges is wrong")
            return False
        return True

    def reference_setting_type0(self, x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo):
        if self.type != 0:
            print("the type of the robots is not 0")
            return
        self.x_mr = x_mr     # 设定轨迹的x坐标
        self.y_mr = y_mr     # 设定轨迹的y坐标
        self.v_mr = v_mr     # 设定轨迹的速度大小
        theta_v = math.atan2(y_mr_dot, x_mr_dot)
        self.v_mr_theta = theta_v    # 设定轨迹的切线角度，即速度的方向角
        if self.status == "working":
            self.theta_mr, _ = theta_mod(
                theta1=theta_mr, theta0=self.theta_mr)     # 设定编队整体的方向角度
        cos_theta_mr = math.cos(self.theta_mr)
        sin_theta_mr = math.sin(self.theta_mr)
        self.w_mr = w_revo+w_rotat  # 设定编队整体的角速度
        beta = beta_rotat+beta_revo
        self.v_mr_theta = theta_v
        self.pos_r = Position(x=self.x_mr, y=self.y_mr, theta=self.v_mr_theta)
        kappa = w_revo/v_mr
        self.kappa_r = kappa
        v_r = np.zeros((1, self.Nrb))
        for i in range(self.Nrb):
            x_i_dot1 = -self.w_mr*(sin_theta_mr*self.init_positions[i].x +
                                   cos_theta_mr*self.init_positions[i].y)+x_mr_dot
            y_i_dot1 = self.w_mr*(cos_theta_mr*self.init_positions[i].x -
                                  sin_theta_mr*self.init_positions[i].y)+y_mr_dot
            x_i_dot2 = -(beta*sin_theta_mr+self.w_mr**2*cos_theta_mr)*self.init_positions[i].x - \
                (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr) * \
                self.init_positions[i].y + x_mr_dot2
            y_i_dot2 = (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr)*self.init_positions[i].x - \
                (beta*sin_theta_mr+self.w_mr**2*cos_theta_mr) * \
                self.init_positions[i].y + y_mr_dot2
            v_ir = math.sqrt(x_i_dot1**2+y_i_dot1**2)
            # v_r[0][i] = v_ir
            w_ir = (y_i_dot2*x_i_dot1-x_i_dot2*y_i_dot1)/v_ir**2

            if self.status == "working":
                self.theta_ir[0][i], _ = theta_mod(theta1=math.atan2(
                    y_i_dot1, x_i_dot1), theta0=self.theta_ir[0][i])
            else:
                self.theta_ir[0][i] = math.atan2(y_i_dot1, x_i_dot1)

            self.v_ir[0][i] = v_ir
            self.w_ir[0][i] = w_ir
            self.u_r[0][i] = self.v_ir[0][i]
            self.u_r[0][i+self.Nrb] = self.w_ir[0][i]

            for k in self.nodes_ht[i]:
                self.theta_ir_theta_mr_phi_kr[i][k] = self.theta_ir[0][i] - \
                    self.theta_mr - self.phi_kr[0][k]
                self.sin_theta_ir_theta_mr_phi_kr[i][k] = math.sin(
                    self.theta_ir_theta_mr_phi_kr[i][k])
                self.cos_theta_ir_theta_mr_phi_kr[i][k] = math.cos(
                    self.theta_ir_theta_mr_phi_kr[i][k])
            self.sin_theta_ir[0][i] = math.sin(self.theta_ir[0][i])
            self.cos_theta_ir[0][i] = math.cos(self.theta_ir[0][i])
            for k in range(self.Neg):
                self.theta_mr_dot1 += (self.v_ir[0][self.edges[k][0]] * self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][0]] -
                                       self.v_ir[0][self.edges[k][1]] * self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][1]])/self.d_kr[0][k]
            self.theta_mr_dot1 /= self.Neg
        self.X_ref = np.hstack((
            np.array([[self.x_mr, self.y_mr, self.theta_mr]]
                     ), self.d_kr, self.theta_ir
        ))
        self.last_tr_features = [x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2,
                                 theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo]

    def setting_reference_brief(self, x_mr, y_mr, v_mr, a_tau_mr, theta_mr, w_mr, beta_mr, theta_vr, kappa, kappa_dot):
        self.kappa_r = kappa
        x_mr_dot = v_mr*math.cos(theta_vr)
        y_mr_dot = v_mr*math.sin(theta_vr)
        a_n_mr = v_mr**2*kappa
        theta_tau_m = theta_vr
        theta_n_m = theta_vr+math.pi/2*np.sign(kappa)
        x_mr_dot2 = math.cos(theta_tau_m)*a_tau_mr+math.cos(theta_n_m)*a_n_mr
        y_mr_dot2 = math.sin(theta_tau_m)*a_tau_mr+math.sin(theta_n_m)*a_n_mr
        w_c = v_mr*kappa
        w_revo = w_c
        w_rotat = w_mr-w_c
        beta_revo = math.sqrt(x_mr_dot2**2+y_mr_dot2**2)*kappa+v_mr*kappa_dot
        beta_rotat = beta_mr-w_c**2
        self.reference_setting(x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2,
                               y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo)

    def setting_reference_tr(self, X_mr, Y_mr, THETA_mr, **kwargs):
        # X_mr, Y_mr, THETA_mr: 参考轨迹上的5个点，前两个为历史轨迹，第3个为下一时刻的参考点，4、5为之后的参考位置
        # 根据X_mr, Y_mr计算参考轨迹上的位置，位置一阶导数，二阶导数，角度，速度，加速度，曲率，曲率导数，自传角速度，自传角加速度，旋转角速度，旋转角加速度
        scale = kwargs.get('scale', 1)
        V_mr = kwargs.get('V_mr', -100)
        offset = kwargs.get('offset', 0)
        if V_mr != -100 and -speed_v_max < V_mr < speed_v_max:
            scale = V_mr/math.sqrt((X_mr[2]-X_mr[1])
                                   ** 2+(Y_mr[2]-Y_mr[1])**2)*self.T
            if scale > 1:
                scale = 1
        if scale != 1 or offset != 0:
            # 对X_mr, Y_mr进行插值计算
            t0 = np.linspace(start=0, stop=4, num=5)*self.T
            t1 = t0*scale+offset*self.T
            f = interpolate.interp1d(t0, X_mr, kind='cubic')
            X_mr = f(t1)
            f = interpolate.interp1d(t0, Y_mr, kind='cubic')
            Y_mr = f(t1)
            f = interpolate.interp1d(t0, THETA_mr, kind='cubic')
            THETA_mr = f(t1)
            self.setting_reference_tr(X_mr, Y_mr, THETA_mr)
            return scale
        # 位置
        x_mr = X_mr[2]
        y_mr = Y_mr[2]
        # 位置一阶导数
        x_mr_dot = (X_mr[4]+X_mr[3]-X_mr[1]-X_mr[0])/self.T/6
        y_mr_dot = (Y_mr[4]+Y_mr[3]-Y_mr[1]-Y_mr[0])/self.T/6
        # 位置二阶导数
        x_mr_dot2 = (X_mr[4]+X_mr[0]-X_mr[1]-X_mr[3])/3/self.T**2
        y_mr_dot2 = (Y_mr[4]+Y_mr[0]-Y_mr[1]-Y_mr[3])/3/self.T**2
        # 角度
        theta_mr = THETA_mr[2]
        for i in range(5):
            THETA_mr[i], _ = theta_mod(theta1=THETA_mr[i], theta0=theta_mr)
        w_all = (THETA_mr[4]+THETA_mr[3]-THETA_mr[1]-THETA_mr[0])/self.T/6
        beta_all = (THETA_mr[4]+THETA_mr[0] -
                    THETA_mr[1]-THETA_mr[3])/3/self.T**2
        # 速度
        v_mr = math.sqrt(x_mr_dot**2+y_mr_dot**2)
        # 加速度
        a_tau_mr = math.sqrt(x_mr_dot2**2+y_mr_dot2**2)
        # 曲率
        kappa = cal_kappa(X_mr[0:3], Y_mr[0:3])
        # 曲率变化率
        kappa1 = cal_kappa(X_mr[2:5], Y_mr[2:5])
        kappa_dot = (kappa1-kappa)/self.T/2
        # 角速度
        kappa = (x_mr_dot*y_mr_dot2-x_mr_dot2*y_mr_dot)/v_mr**3
        w_revo_r = v_mr*kappa
        w_all = (THETA_mr[2]-THETA_mr[0])/self.T/2
        w_rotat_r = w_all-w_revo_r
        # 角加速度
        beta_all = (THETA_mr[2]-2*THETA_mr[1]+THETA_mr[0])/self.T**2
        beta_revo_r = v_mr*kappa_dot+a_tau_mr*kappa
        beta_rotat_r = beta_all-beta_revo_r
        self.reference_setting(x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2,
                               y_mr_dot2, theta_mr, v_mr, w_rotat_r, beta_rotat_r, w_revo_r, beta_revo_r)
        return scale

    def reference_setting_type1(self, x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo):
        if self.type != 1:
            return
        # type1 全向移动机器人
        self.x_mr = x_mr     # 设定轨迹的x坐标
        self.y_mr = y_mr     # 设定轨迹的y坐标
        self.v_mr = v_mr     # 设定轨迹的速度大小
        theta_v = math.atan2(y_mr_dot, x_mr_dot)
        self.v_mr_theta = theta_v    # 设定轨迹的切线角度，即速度的方向角
        if self.status == "working":
            self.theta_mr, _ = theta_mod(
                theta1=theta_mr, theta0=self.theta_mr)     # 设定编队整体的方向角度
        cos_theta_mr = math.cos(self.theta_mr)
        sin_theta_mr = math.sin(self.theta_mr)
        self.w_mr = w_revo+w_rotat  # 设定编队整体的角速度
        beta = beta_rotat+beta_revo
        self.v_mr_theta = theta_v
        self.pos_r = Position(x=self.x_mr, y=self.y_mr, theta=self.v_mr_theta)
        kappa = w_revo/v_mr
        self.kappa_r = kappa
        for i in range(self.Nrb):
            x_i_dot1 = -self.w_mr*(sin_theta_mr*self.init_positions[i].x +
                                   cos_theta_mr*self.init_positions[i].y)+x_mr_dot
            y_i_dot1 = self.w_mr*(cos_theta_mr*self.init_positions[i].x -
                                  sin_theta_mr*self.init_positions[i].y)+y_mr_dot
            x_i_dot2 = -(beta*sin_theta_mr+self.w_mr**2*cos_theta_mr)*self.init_positions[i].x - \
                (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr) * \
                self.init_positions[i].y + x_mr_dot2
            y_i_dot2 = (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr)*self.init_positions[i].x - \
                (beta*sin_theta_mr+self.w_mr**2*cos_theta_mr) * \
                self.init_positions[i].y + y_mr_dot2
            v_ir = math.sqrt(x_i_dot1**2+y_i_dot1**2)
            # self.v_ir[0][i] = v_ir
            theta_i = math.atan2(y_i_dot1, x_i_dot1)
            w_ir = (y_i_dot2*x_i_dot1-x_i_dot2*y_i_dot1)/v_ir**2
            # theta_i = 0
            # w_ir = 0
            vx_ir = x_i_dot1*math.cos(theta_i) + y_i_dot1*math.sin(theta_i)
            vy_ir = -x_i_dot1*math.sin(theta_i) + y_i_dot1*math.cos(theta_i)
            if self.status == "working":
                self.theta_ir[0][i], _ = theta_mod(
                    theta1=theta_i, theta0=self.theta_ir[0][i])
            else:
                self.theta_ir[0][i] = theta_i

            self.vx_ir[0][i] = vx_ir
            self.vy_ir[0][i] = vy_ir
            self.w_ir[0][i] = w_ir
            self.u_r[0][i] = self.vx_ir[0][i]
            self.u_r[0][i+self.Nrb] = self.vy_ir[0][i]
            self.u_r[0][i+self.Nrb*2] = self.w_ir[0][i]

            self.sin_theta_ir[0][i] = math.sin(self.theta_ir[0][i])
            self.cos_theta_ir[0][i] = math.cos(self.theta_ir[0][i])
        for k in range(self.Neg):
            self.cos_theta_m_phi_kr[0][k] = math.cos(
                self.theta_mr+self.phi_kr[0][k])
            self.sin_theta_m_phi_kr[0][k] = math.sin(
                self.theta_mr+self.phi_kr[0][k])
        self.X_ref = np.hstack((
            np.array([[self.x_mr, self.y_mr, self.theta_mr]]
                     ), self.d_kr, self.theta_ir
        ))
        self.last_tr_features = [x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2,
                                 theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo]

    def reference_setting_type2(self, x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo):
        if self.type != 2:
            return
        self.x_mr = x_mr     # 设定轨迹的x坐标
        self.y_mr = y_mr     # 设定轨迹的y坐标
        self.v_mr = v_mr     # 设定轨迹的速度大小
        theta_v = math.atan2(y_mr_dot, x_mr_dot)
        self.v_mr_theta = theta_v    # 设定轨迹的切线角度，即速度的方向角
        if self.status == "working":
            self.theta_mr, _ = theta_mod(
                theta1=theta_mr, theta0=self.theta_mr)     # 设定编队整体的方向角度
        cos_theta_mr = math.cos(self.theta_mr)
        sin_theta_mr = math.sin(self.theta_mr)
        self.w_mr = w_revo+w_rotat  # 设定编队整体的角速度
        beta = beta_rotat+beta_revo
        self.v_mr_theta = theta_v
        self.pos_r = Position(x=self.x_mr, y=self.y_mr, theta=self.v_mr_theta)
        kappa = w_revo/v_mr
        self.kappa_r = kappa
        for i in range(self.Nrb):
            x_i_dot1 = -self.w_mr*(sin_theta_mr*self.init_positions[i].x +
                                   cos_theta_mr*self.init_positions[i].y)+x_mr_dot
            y_i_dot1 = self.w_mr*(cos_theta_mr*self.init_positions[i].x -
                                  sin_theta_mr*self.init_positions[i].y)+y_mr_dot
            x_i_dot2 = -(beta*sin_theta_mr+self.w_mr**2*cos_theta_mr)*self.init_positions[i].x - \
                (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr) * \
                self.init_positions[i].y + x_mr_dot2
            y_i_dot2 = (beta*cos_theta_mr-self.w_mr**2*sin_theta_mr)*self.init_positions[i].x - \
                (beta*sin_theta_mr+self.w_mr**2*cos_theta_mr) * \
                self.init_positions[i].y + y_mr_dot2
            v_ir = math.sqrt(x_i_dot1**2+y_i_dot1**2)
            w_ir = (y_i_dot2*x_i_dot1-x_i_dot2*y_i_dot1)/v_ir**2

            if self.status == "working":
                self.theta_ir[0][i], _ = theta_mod(theta1=math.atan2(
                    y_i_dot1, x_i_dot1), theta0=self.theta_ir[0][i])
            else:
                self.theta_ir[0][i] = math.atan2(y_i_dot1, x_i_dot1)

            self.v_ir[0][i] = v_ir
            self.phi_ir[0][i] = math.atan2(w_ir*self.robots[i].L, v_ir)
            self.u_r[0][i] = self.v_ir[0][i]
            self.u_r[0][i+self.Nrb] = self.phi_ir[0][i]
            for k in self.nodes_ht[i]:
                self.theta_ir_theta_mr_phi_kr[i][k] = self.theta_ir[0][i] - \
                    self.theta_mr - self.phi_kr[0][k]
                self.sin_theta_ir_theta_mr_phi_kr[i][k] = math.sin(
                    self.theta_ir_theta_mr_phi_kr[i][k])
                self.cos_theta_ir_theta_mr_phi_kr[i][k] = math.cos(
                    self.theta_ir_theta_mr_phi_kr[i][k])
            self.sin_theta_ir[0][i] = math.sin(self.theta_ir[0][i])
            self.cos_theta_ir[0][i] = math.cos(self.theta_ir[0][i])
            for k in range(self.Neg):
                self.theta_mr_dot1 += (self.v_ir[0][self.edges[k][0]] *
                                       self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][0]] -
                                       self.v_ir[0][self.edges[k][1]] *
                                       self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][1]])/self.d_kr[0][k]
            self.theta_mr_dot1 /= self.Neg
        self.X_ref = np.hstack((
            np.array([[self.x_mr, self.y_mr, self.theta_mr]]
                     ), self.d_kr, self.theta_ir
        ))
        self.last_tr_features = [x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2,
                                 theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo]

    def reference_setting(self, x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo):
        if self.type == 0:
            return self.reference_setting_type0(x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo)
        elif self.type == 1:
            return self.reference_setting_type1(x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo)
        elif self.type == 2:
            return self.reference_setting_type2(x_mr, y_mr, x_mr_dot, y_mr_dot, x_mr_dot2, y_mr_dot2, theta_mr, v_mr, w_rotat, beta_rotat, w_revo, beta_revo)

    def judge_reference(self, method="cone intersection", **kwargs):
        # cone intersection
        if method == "cone intersection":
            # calculate rotaion center
            angle_cone = kwargs.get("angle_cone", math.pi/12)
            cos_angle_cone = math.cos(angle_cone)
            flag = True
            theta_errors = []
            if np.abs(self.w_mr) < 1e-8:
                # center = Position(x=float["inf"], y=float["inf"])
                theta_zero = self.v_mr_theta
                for i in range(self.Nrb):
                    theta_error, error_flag = theta_mod(
                        theta1=self.robots[i].pos.theta, theta0=theta_zero, error_max=math.pi)
                    theta_errors.append(theta_error)
                theta_errors -= theta_zero
                if np.abs(np.max(theta_errors)) > angle_cone:
                    flag = False
            else:
                R0 = self.v_mr/self.w_mr
                center = Position(x=self.x_mr + R0 * math.cos(self.v_mr_theta + np.sign(R0) * math.pi / 2),
                                  y=self.y_mr + R0 * math.sin(self.v_mr_theta + np.sign(R0) * math.pi / 2))
                for rb in self.robots:
                    theta_XO = math.atan2(center.y-rb.pos.y, center.x-rb.pos.x)
                    # theta_sup = rb.pos.theta-angle_cone-math.pi/2
                    # theta_inf = rb.pos.theta+angle_cone-math.pi/2
                    theta_error, error_flag = theta_mod(
                        theta_XO-rb.pos.theta+math.pi/2, theta0=0, error_max=math.pi)
                    theta_errors.append(theta_error)
                    if np.abs(math.cos(theta_XO-rb.pos.theta+math.pi/2)) < cos_angle_cone:
                        flag = False
            return flag, theta_errors
        # eudlidean distance
        elif method == "eudlidean distance":
            d_min = kwargs.get("d_min", 0.1)
            flag = True
            for i in self.Nrb:
                if self.robots[i].pos.distance(
                        other=Position.trans(self=self.init_positions[i],
                                             translate_x=self.x_mr, translate_y=self.y_mr, rotation_theta=self.theta_mr),
                        option="euclidean_Q") > d_min:
                    flag = False
                    break
            return flag
        # line intersection
        # elif method == "line intersection":
        #     d_min=kwargs["line_d_min"]
        #     flag=True
        #     for rb in self.robots:
        #         if (math.cos(rb.pos.theta-math.pi/2)*())
        #     pass

    def adjust_theta(self, theta_errors, **kwargs):
        kp = kwargs.get("kp", 0.0)
        dec_rate = kwargs.get("dec_rate", 0.8)
        # kd=kwargs.get("kd",0.0)
        # ki=kwargs.get("ki",0.0)
        for i in range(self.Nrb):
            self.robots[i].speed.pub_v *= dec_rate
            self.robots[i].speed.pub_w -= kp*theta_errors[i]
            self.robots[i].speed.pub_u[0] = self.robots[i].speed.pub_v
            self.robots[i].speed.pub_u[1] = self.robots[i].speed.pub_w
            self.u_r[0][i] = self.robots[i].speed.pub_v
            self.u_r[0][i+self.Nrb] = self.robots[i].speed.pub_w

    def calculate_matrix_jx(self):
        jx13 = np.array([[0]])
        jx14 = np.zeros((1, self.Neg))
        jx15 = np.zeros((1, self.Nrb))
        jx23 = np.array([[0]])
        jx24 = np.zeros((1, self.Neg))
        jx25 = np.zeros((1, self.Nrb))
        jx33 = np.array([[0]])
        jx34 = np.zeros((1, self.Neg))
        jx35 = np.zeros((1, self.Nrb))
        for i in range(self.Nrb):
            jx15[0][i] = -1 / self.Nrb * \
                self.v_ir[0][i] * self.sin_theta_ir[0][i]
            jx25[0][i] = 1 / self.Nrb * \
                self.v_ir[0][i] * self.cos_theta_ir[0][i]
            for k in self.nodes_head[i]:
                jx35[0][i] += self.v_ir[0][i] * \
                    self.cos_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.Neg / self.d_kr[0][k]
            for k in self.nodes_tail[i]:
                jx35[0][i] -= self.v_ir[0][i] * \
                    self.cos_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.Neg / self.d_kr[0][k]
        jx43 = np.zeros((self.Neg, 1))
        jx45 = np.zeros((self.Neg, self.Nrb))
        for k in range(self.Neg):
            jx33[0][0] += (-self.v_ir[0][self.edges[k][0]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] +
                           self.v_ir[0][self.edges[k][1]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]) / \
                self.d_kr[0][k]
            jx34[0][k] = (-self.v_ir[0][self.edges[k][0]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] +
                          self.v_ir[0][self.edges[k][1]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]) / (
                self.d_kr[0][k] ** 2) / self.Neg

            num_h = self.v_ir[0][self.edges[k][0]] * \
                self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k]
            num_t = self.v_ir[0][self.edges[k][1]] * \
                self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]
            jx43[k][0] = num_h - num_t
            jx45[k][self.edges[k][0]] = -num_h
            jx45[k][self.edges[k][1]] = num_t
        jx33[0][0] /= self.Neg

        if (np.abs(self.sum_init_x) >= 1e-8) or (np.abs(self.sum_init_y) >= 0):
            sum_cosx_siny = self.cos_theta_m*self.sum_init_x+self.sin_theta_m*self.sum_init_y
            sum_sinx_cosy = self.sin_theta_m*self.sum_init_x+self.cos_theta_m*self.sum_init_y
            jx13[0][0] = (self.theta_mr_dot1*sum_cosx_siny +
                          jx33[0][0]*sum_sinx_cosy)/self.Nrb
            jx23[0][0] = -(-self.theta_mr_dot1*sum_sinx_cosy +
                           jx33[0][0]*sum_cosx_siny)/self.Nrb
            for k in range(self.Neg):
                jx14[0][k] = jx34[0][k]*sum_sinx_cosy
                jx24[0][k] = -jx34[0][k]*sum_cosx_siny

        return np.vstack((
            np.hstack((np.zeros((1, 2)), jx13, jx14, jx15)),
            np.hstack((np.zeros((1, 2)), jx23, jx24, jx25)),
            np.hstack((np.zeros((1, 2)), jx33, jx34, jx35)),
            np.hstack((np.zeros((self.Neg, 2)), jx43,
                      np.zeros((self.Neg, self.Neg)), jx45)),
            np.zeros((self.Nrb, self.Nst))
        ))

    def calculate_matrix_ju(self):
        ju11 = np.zeros((1, self.Nrb))
        ju21 = np.zeros((1, self.Nrb))
        ju31 = np.zeros((1, self.Nrb))
        ju41 = np.zeros((self.Neg, self.Nrb))
        ju52 = np.eye(self.Nrb, self.Nrb)
        for i in range(self.Nrb):
            ju11[0][i] = self.cos_theta_ir[0][i] / self.Nrb
            ju21[0][i] = self.sin_theta_ir[0][i] / self.Nrb
            for k in self.nodes_head[i]:
                ju31[0][i] += self.sin_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
            for k in self.nodes_tail[i]:
                ju31[0][i] -= self.sin_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
            ju31[0][i] /= self.Neg
        for k in range(self.Neg):
            ju41[k][self.edges[k][0]
                    ] = self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k]
            ju41[k][self.edges[k][1]] = - \
                self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]
        return np.vstack((
            np.hstack((ju11, np.zeros((1, self.Nrb)))),
            np.hstack((ju21, np.zeros((1, self.Nrb)))),
            np.hstack((ju31, np.zeros((1, self.Nrb)))),
            np.hstack((ju41, np.zeros((self.Neg, self.Nrb)))),
            np.hstack((np.zeros((self.Nrb, self.Nrb)), ju52))
        ))

    def calculate_matrix_jx_type1(self):
        jx13 = np.array([[0]])
        jx14 = np.zeros((1, self.Neg))
        jx15 = np.zeros((1, self.Nrb))
        jx23 = np.array([[0]])
        jx24 = np.zeros((1, self.Neg))
        jx25 = np.zeros((1, self.Nrb))
        jx33 = np.array([[0]])
        jx34 = np.zeros((1, self.Neg))
        jx35 = np.zeros((1, self.Nrb))
        jx35 = np.zeros((1, self.Nrb))

        for i in range(self.Nrb):
            jx15[0][i] = -1/self.Nrb * (self.vx_ir[0][i]*self.sin_theta_ir[0]
                                        [i]+self.vy_ir[0][i]*self.cos_theta_ir[0][i])
            jx25[0][i] = 1/self.Nrb*(self.vx_i[0][i]*self.cos_theta_ir[0]
                                     [i]-self.vy_ir[0][i]*self.sin_theta_ir[0][i])
            for k in self.nodes_head[i]:
                jx35[0][i] += (self.vx_ir[0][i]*self.cos_theta_ir_theta_mr_phi_kr[i][k]-self.vy_ir[0]
                               [i]*self.sin_theta_ir_theta_mr_phi_kr[i][k])/self.Neg/self.d_kr[0][k]
            for k in self.nodes_tail[i]:
                jx35[0][i] -= (self.vx_ir[0][i]*self.cos_theta_ir_theta_mr_phi_kr[i][k]-self.vy_ir[0]
                               [i]*self.sin_theta_ir_theta_mr_phi_kr[i][k])/self.Neg/self.d_kr[0][k]
        jx43 = np.zeros((self.Neg, 1))
        jx45 = np.zeros((self.Neg, self.Nrb))
        # vx_i = self.vx_ir[0][i]
        # vy_i = self.vy_ir[0][i]
        # jx13[0][0] += -vx_i*self.sin_theta_ir[0][i] - \
        #     vy_i*self.cos_theta_ir[0][i]
        # jx23[0][0] += vx_i*self.cos_theta_ir[0][i] - \
        #     vy_i*self.sin_theta_ir[0][i]
        jx13[0][0] /= self.Nrb
        jx23[0][0] /= self.Nrb
        for k in range(self.Neg):
            # Dvx_cos = (self.vx_ir[0][self.edges[k][0]]-self.vx_ir[0][self.edges[k][1]]) *\
            #     self.cos_theta_m_phi_kr[0][self.edges[k][1]]
            # Dvx_sin = (self.vx_ir[0][self.edges[k][0]]-self.vx_ir[0][self.edges[k][1]]) *\
            #     self.sin_theta_m_phi_kr[0][self.edges[k][1]]
            # Dvy_cos = (self.vy_ir[0][self.edges[k][0]]-self.vy_ir[0][self.edges[k][1]]) *\
            #     self.cos_theta_m_phi_kr[0][self.edges[k][1]]
            # Dvy_sin = (self.vy_ir[0][self.edges[k][0]]-self.vy_ir[0][self.edges[k][1]]) *\
            #     self.sin_theta_m_phi_kr[0][self.edges[k][1]]
            # jx33[0][0] -= (Dvx_cos+Dvy_sin) / self.d_kr[0][k]
            # jx34[0][k] = (Dvx_sin-Dvy_cos) / (self.d_kr[0][k] ** 2) / self.Neg
            # jx43[k][0] = -Dvx_sin+Dvy_cos
            jx33[0][0] += (-self.vx_ir[0][self.edges[k][0]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] +
                           self.vx_ir[0][self.edges[k][1]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k] -
                           self.vy_ir[0][self.edges[k][0]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] +
                           self.vy_ir[0][self.edges[k][1]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]) / \
                self.d_kr[0][k]
            jx34[0][k] = (-self.vx_ir[0][self.edges[k][0]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] +
                          self.vx_ir[0][self.edges[k][1]] * self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k] +
                          self.vy_ir[0][self.edges[k][0]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] -
                          self.vy_ir[0][self.edges[k][1]] * self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]) / (
                self.d_kr[0][k] ** 2) / self.Neg

            num_h = self.vx_ir[0][self.edges[k][0]] * \
                self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k] + \
                self.vy_ir[0][self.edges[k][0]] * \
                self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k]
            num_t = self.vx_ir[0][self.edges[k][1]] * \
                self.sin_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k] + \
                self.vx_ir[0][self.edges[k][1]] * \
                self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]
            jx43[k][0] = num_h - num_t
            jx45[k][self.edges[k][0]] = -num_h
            jx45[k][self.edges[k][1]] = num_t
        jx33[0][0] /= self.Neg

        return np.vstack((
            np.hstack((np.zeros((1, 2)), jx13, jx14, jx15)),
            np.hstack((np.zeros((1, 2)), jx23, jx24, jx25)),
            np.hstack((np.zeros((1, 2)), jx33, jx34, jx35)),
            np.hstack((np.zeros((self.Neg, 2)), jx43,
                      np.zeros((self.Neg, self.Neg)), jx45)),
            np.zeros((self.Nrb, self.Nst))
        ))

    def calculate_matrix_ju_type1(self):
        ju11 = np.zeros((1, self.Nrb))
        ju12 = np.zeros((1, self.Nrb))
        ju21 = np.zeros((1, self.Nrb))
        ju22 = np.zeros((1, self.Nrb))
        ju31 = np.zeros((1, self.Nrb))
        ju32 = np.zeros((1, self.Nrb))
        ju41 = np.zeros((self.Neg, self.Nrb))
        ju42 = np.zeros((self.Neg, self.Nrb))
        ju53 = np.eye(self.Nrb, self.Nrb)
        for i in range(self.Nrb):
            ju11[0][i] = self.cos_theta_ir[0][i] / self.Nrb
            ju12[0][i] = -self.sin_theta_ir[0][i] / self.Nrb
            ju21[0][i] = self.sin_theta_ir[0][i] / self.Nrb
            ju22[0][i] = self.cos_theta_ir[0][i] / self.Nrb
            for k in self.nodes_head[i]:
                # theta_c/vx
                ju31[0][i] += self.sin_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
                # theta_c/vy
                ju32[0][i] += self.cos_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
            for k in self.nodes_tail[i]:
                # theta_c/vx
                ju31[0][i] -= self.sin_theta_ir_theta_mr_phi_kr[0][i] / \
                    self.d_kr[0][i]
                # theta_c/vy
                ju32[0][i] -= self.cos_theta_ir_theta_mr_phi_kr[0][i] / \
                    self.d_kr[0][i]
            ju31[0][i] /= self.Neg
            ju32[0][i] /= self.Neg
        for k in range(self.Neg):
            ju41[k][self.edges[k][0]
                    ] = self.cos_theta_ir_theta_mr_phi_kr[0][self.edges[k][0]]
            ju41[k][self.edges[k][1]] = - \
                self.cos_theta_ir_theta_mr_phi_kr[0][self.edges[k][1]]
            ju42[k][self.edges[k][0]] = - \
                self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][0]]
            ju42[k][self.edges[k][1]
                    ] = self.sin_theta_ir_theta_mr_phi_kr[0][self.edges[k][1]]
        return np.vstack((
            np.hstack((ju11, ju12, np.zeros((1, self.Nrb)))),
            np.hstack((ju21, ju22, np.zeros((1, self.Nrb)))),
            np.hstack((ju31, ju32, np.zeros((1, self.Nrb)))),
            np.hstack((ju41, ju42, np.zeros((self.Neg, self.Nrb)))),
            np.hstack((np.zeros((self.Nrb, self.Nrb*2)), ju53))
        ))

    def calculate_matrix_jx_type2(self):
        return self.calculate_matrix_jx()

    def calculate_matrix_ju_type2(self):
        ju11 = np.zeros((1, self.Nrb))
        ju21 = np.zeros((1, self.Nrb))
        ju31 = np.zeros((1, self.Nrb))
        ju41 = np.zeros((self.Neg, self.Nrb))
        ju51 = np.zeros((self.Nrb, self.Nrb))
        ju52 = np.zeros((self.Nrb, self.Nrb))
        for i in range(self.Nrb):
            ju11[0][i] = self.cos_theta_ir[0][i] / self.Nrb
            ju21[0][i] = self.sin_theta_ir[0][i] / self.Nrb
            for k in self.nodes_head[i]:
                ju31[0][i] += self.sin_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
            for k in self.nodes_tail[i]:
                ju31[0][i] -= self.sin_theta_ir_theta_mr_phi_kr[i][k] / \
                    self.d_kr[0][k]
            ju31[0][i] /= self.Neg
            ju51[i][i] = math.tan(self.phi_ir[0][i])/self.robots[i].L
            ju52[i][i] = self.v_ir[0][i] / \
                self.robots[i].L/(1+(self.phi_ir[0][i])**2)
        for k in range(self.Neg):
            ju41[k][self.edges[k][0]
                    ] = self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][0]][k]
            ju41[k][self.edges[k][1]] = - \
                self.cos_theta_ir_theta_mr_phi_kr[self.edges[k][1]][k]
        return np.vstack((
            np.hstack((ju11, np.zeros((1, self.Nrb)))),
            np.hstack((ju21, np.zeros((1, self.Nrb)))),
            np.hstack((ju31, np.zeros((1, self.Nrb)))),
            np.hstack((ju41, np.zeros((self.Neg, self.Nrb)))),
            np.hstack((ju51, ju52))
        ))

    def calculate_matrix_a(self, sample_time=0.1):
        if self.type == 0:
            return self.calculate_matrix_jx() * sample_time + np.eye(self.Nst)
        elif self.type == 1:
            return self.calculate_matrix_jx_type1() * sample_time + np.eye(self.Nst)
        elif self.type == 2:
            return self.calculate_matrix_jx_type2() * sample_time + np.eye(self.Nst)

    def calculate_matrix_b(self, sample_time=0.1):
        if self.type == 0:
            return self.calculate_matrix_ju() * sample_time
        elif self.type == 1:
            return self.calculate_matrix_ju_type1() * sample_time
        elif self.type == 2:
            return self.calculate_matrix_ju_type2() * sample_time

    def mpc_control(self, Np=5, T=None, d_err_max=0.1):
        if not T and self.T > 0:
            T = self.T
            for rb in self.robots:
                rb.T = T
        elif not T and self.T <= 0:
            T = 0.1
            self.T = T
            for rb in self.robots:
                rb.T = T
        elif T <= 0:
            print("T is not defined")
            return
        if self.type == 0:
            ub = np.array([speed_v_max] * self.Nrb + [speed_w_max] * self.Nrb)
            lb = np.array([-speed_v_max] * self.Nrb +
                          [-speed_w_max] * self.Nrb)
        elif self.type == 1:
            ub = np.array([speed_v_max, speed_v_max] *
                          self.Nrb + [speed_w_max] * self.Nrb)
            lb = np.array([-speed_v_max, -speed_v_max] *
                          self.Nrb + [-speed_w_max] * self.Nrb)
        elif self.type == 2:
            ub = np.array([speed_phi_max] * self.Nrb +
                          [speed_w_max] * self.Nrb)
            lb = np.array([-speed_phi_max] * self.Nrb +
                          [-speed_w_max] * self.Nrb)
        X_tilde = self.X_real - self.X_ref
        for i in range(self.Nst):
            if i < 2:
                if np.abs(X_tilde[0][i]) > 0.25:
                    print(i+1, "x_c or y_c is out of range")
                    print(self.X_ref[0][i], self.X_real[0][i])
                    self.mpcOK = False
            elif i >= 3 and i < 3+self.Neg:
                if np.abs(X_tilde[0][i]) > d_err_max:
                    print("d_", i-2, " is out of range", X_tilde[0][i])
                    self.mpcOK = False
            elif i == 2 or i >= 3+self.Neg:
                if np.abs(X_tilde[0][i]) > theta_error_max:
                    print("theta_c or theta_k is out of range", i)
                    print("theta_r:", self.X_ref[0][i])
                    print("theta:", self.X_real[0][i])
                    self.mpcOK = False
        # for i in range(3 + self.Neg, self.Nst):
        #     X_tilde[0][i] = math.asin(math.sin(X_tilde[0][i]))

        Q = np.diag(([weight["x_m"], weight["y_m"], weight["theta_m"]] + [weight["d_k"]] * self.Neg + [
            weight["theta_k"]] * self.Nrb) * Np)
        if self.type == 0:
            R = np.diag(([weight["v_k"]] * self.Nrb +
                        [weight["w_k"]] * self.Nrb) * Np)
        elif self.type == 1:
            R = np.diag(([weight["v_x_k"]] * self.Nrb + [weight["v_y_k"]]
                        * self.Nrb + [weight["w_k"]] * self.Nrb) * Np)
        elif self.type == 2:
            R = np.diag(([weight["v_k"]] * self.Nrb +
                        [weight["phi_k"]] * self.Nrb) * Np)
        a = self.calculate_matrix_a(sample_time=T)
        b = self.calculate_matrix_b(sample_time=T)
        A = a
        B = b
        for i in range(Np - 1):
            B = np.hstack((B, np.zeros(b.shape)))
        aa = [a]
        bb = [b]
        for i in range(Np - 1):
            aa.append(np.dot(aa[i], a))
            bb.append(np.dot(a, bb[i]))
        for i in range(Np - 1):
            A = np.vstack((A, aa[i + 1]))
            BB = bb[i + 1]
            for j in range(Np - 1):
                if j <= i:
                    BB = np.hstack((BB, bb[i - j]))
                else:
                    BB = np.hstack((BB, np.zeros(b.shape)))
            B = np.vstack((B, BB))
        H = 2 * (np.dot(np.dot(B.T, Q), B) + R)
        f = 2 * np.dot(np.dot(np.dot(B.T, Q), A), X_tilde.T)
        lb = lb - self.u_r
        ub = ub - self.u_r
        LB = lb
        UB = ub
        for i in range(Np - 1):
            LB = np.hstack((LB, lb))
            UB = np.hstack((UB, ub))
        E0 = np.hstack((np.zeros((self.Neg, 3)), np.eye(
            self.Neg), np.zeros((self.Neg, self.Nrb))))
        E = np.hstack((E0, np.zeros((self.Neg, (Np - 1) * self.Nst))))
        for i in range(Np - 1):
            E = np.vstack((E, np.hstack((np.zeros((self.Neg, (i + 1) * self.Nst)),
                                         E0, np.zeros((self.Neg, (Np - i - 2) * self.Nst))))))
        Aneq = np.dot(np.vstack((E, -E)), B)
        b0 = np.dot(np.dot(E, A), X_tilde.T)
        bneq = np.vstack((np.ones((self.Neg * Np, 1)) * d_err_max -
                         b0, np.ones((self.Neg * Np, 1)) * d_err_max + b0))
        X = solve_qp(H=H, f=f, Aneq=Aneq, bneq=bneq, lb=LB, ub=UB)
        for i in range(self.Nrb):
            if self.type == 0:
                self.u[0][i] = self.u_r[0][i] + X[i]
                self.u[0][i + self.Nrb] = self.u_r[0][i +
                                                      self.Nrb] + X[i + self.Nrb]
                if np.abs(self.robots[i].speed.pub_v-self.u[0][i])/T > speed_a_max:
                    print("a: ", (self.robots[i].speed.pub_v-self.u[0][i])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_v = self.u[0][i]
                if np.abs(self.robots[i].speed.pub_w-self.u[0][i + self.Nrb])/T > speed_beta_max:
                    print(
                        "beta: ", (self.robots[i].speed.pub_w-self.u[0][i + self.Nrb])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_w = self.u[0][i + self.Nrb]
            elif self.type == 1:
                self.u[0][i] = self.u_r[0][i] + X[i]
                self.u[0][i + self.Nrb] = self.u_r[0][i +
                                                      self.Nrb] + X[i + self.Nrb]
                self.u[0][i + self.Nrb*2] = self.u_r[0][i +
                                                        self.Nrb*2] + X[i + self.Nrb*2]
                if np.abs(self.robots[i].speed.pub_v_x-self.u[0][i])/T > speed_a_max or np.abs(self.robots[i].speed.pub_v_y-self.u[0][i+self.Nrb])/T > speed_a_max:
                    print("a: ", (self.robots[i].speed.pub_v-self.u[0][i])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_v_x = self.u[0][i]
                    self.robots[i].speed.pub_v_y = self.u[0][i+self.Nrb]
                if np.abs(self.robots[i].speed.pub_w-self.u[0][i + self.Nrb*2])/T > speed_beta_max:
                    print(
                        "beta: ", (self.robots[i].speed.pub_w-self.u[0][i + self.Nrb])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_w = self.u[0][i + self.Nrb*2]
            elif self.type == 2:
                self.u[0][i] = self.u_r[0][i] + X[i]
                self.u[0][i + self.Nrb] = self.u_r[0][i +
                                                      self.Nrb] + X[i + self.Nrb]
                if np.abs(self.robots[i].speed.pub_v-self.u[0][i])/T > speed_a_max:
                    print("a: ", (self.robots[i].speed.pub_v-self.u[0][i])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_v = self.u[0][i]
                if np.abs(self.robots[i].speed.pub_phi-self.u[0][i + self.Nrb])/T > speed_w_max*10000:
                    print(
                        "w: ", (self.robots[i].speed.pub_phi-self.u[0][i + self.Nrb])/T)
                    self.mpcOK = False
                else:
                    self.robots[i].speed.pub_phi = self.u[0][i + self.Nrb]

    def add_mission(self, mission_id, file_name):
        self.missions.update({mission_id: file_name})

    def mission_start(self, mission_id, recode_path="./data/exptest/"):
        # check if the recode_path exists, if not, create it
        if not os.path.exists(recode_path):
            os.makedirs(recode_path)
        # check if the mission_id is in the missions
        if mission_id not in self.missions.keys():
            print("mission_id is not in the missions")
            return
        try:
            reference_path = xlrd.open_workbook(filename=file_name)
        except:
            print("file is not found")
            return

        table = reference_path.sheets()[0]
        # 把table中的数据转化为二维数组
        path_data = np.array([[table.cell_value(i+1, j)
                               for j in range(table.ncols)]for i in range(table.nrows-1)])

        T = path_data[1][0]
        [N, _] = path_data.shape
        N -= 1
        i = 0
        from concurrent.futures import ThreadPoolExecutor
        from apscheduler.schedulers.blocking import BlockingScheduler
        sechduler = BlockingScheduler(TimeoutError='Asia/Shanghai')
        pool_get_positon = ThreadPoolExecutor(max_workers=10)
        pool_publish_speed = ThreadPoolExecutor(max_workers=10)
        pool_save = ThreadPoolExecutor(max_workers=10)
        file_name = self.missions[mission_id]
        # try to open the file, if not, return

        def loop():
            nonlocal i, N, sechduler, pool_get_positon, pool_publish_speed, recode_path
            if i < N:
                info_recv = pool_get_positon.map(
                    Robot.get_position, self.robots)
                for info in info_recv:
                    print("single robot got positon costs", info, 's')
                self.states_update()
                self.reference_setting(x_mr=path_data[i][1], y_mr=path_data[i][2],
                                       x_mr_dot=path_data[i][3], y_mr_dot=path_data[i][4],
                                       x_mr_dot2=path_data[i][5], y_mr_dot2=path_data[i][6],
                                       theta_mr=path_data[i][7], v_mr=path_data[i][8], w_rotat=path_data[i][12], beta_rotat=path_data[i][13], w_revo=path_data[i][14], beta_revo=path_data[i][15])
                self.mpc_control(Np=5, T=T)
                print(self.u_r)
                # print(self.u)
                if self.mpcOK and self.emec == 0:
                    info_recv = pool_publish_speed.map(
                        Robot.publish_speed, self.robots)
                    for info in info_recv:
                        print("single robot publish speed costs", info[1], 's')
                    i += 1
                else:
                    print("reference is not valid")
                    i = N+1
            else:
                self.record_history(recode_path+"RobotSys_exp")
                recode_paths = []
                for rb in self.robots:
                    recode_paths.append(recode_path+"rbobot"+rb.name)
                info_recv = pool_save.map(
                    Robot.record_history, self.robots, recode_paths)
                for info in info_recv:
                    print(info)
                sechduler.remove_all_jobs()
                sechduler.shutdown(wait=False)

        sechduler.add_job(loop, 'interval', seconds=T)
        sechduler.start()

    def mission_simulation(self, mission_id, recode_path="./data/simutest/"):
        file_name = self.missions[mission_id]
        # check if the recode_path exists, if not, create it
        if not os.path.exists(recode_path):
            os.makedirs(recode_path)
        # check if the mission_id is in the missions
        if mission_id not in self.missions.keys():
            print("mission_id is not in the missions")
            return
        # check the file of file_name is exist or not
        try:
            reference_path = xlrd.open_workbook(filename=file_name)
        except:
            print("file is not found")
            return

        table = reference_path.sheets()[0]
        # 把table中的数据转化为二维数组
        path_data = np.array([[table.cell_value(i+1, j)
                               for j in range(table.ncols)]for i in range(table.nrows-1)])

        T = path_data[1][0]
        [N, _] = path_data.shape
        N -= 1
        for i in range(N):
            self.states_update()
            self.reference_setting(x_mr=path_data[i][1], y_mr=path_data[i][2],
                                   x_mr_dot=path_data[i][3], y_mr_dot=path_data[i][4],
                                   x_mr_dot2=path_data[i][5], y_mr_dot2=path_data[i][6],
                                   theta_mr=path_data[i][7], v_mr=path_data[i][8],
                                   w_rotat=path_data[i][12], beta_rotat=path_data[i][13],
                                   w_revo=path_data[i][14], beta_revo=path_data[i][15])
            self.mpc_control(Np=5, T=T)
            for rb in self.robots:
                rb.simu_predict(delta_t=T)
        self.record_history(recode_path+"RobotSys_simu")
        for rb in self.robots:
            rb.record_history(recode_path+"rbobot"+rb.name)

