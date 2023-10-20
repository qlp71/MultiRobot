from robot import Robot
from robot import RobotsSystem
from robot import Position
from robot import Speed
import math

if __name__ == '__main__':
    sq2 = math.sqrt(2)/2
    type = 0
    R1 = Robot(ip_addr="192.168.1.101", pos=Position(
        x=sq2, y=sq2-10/3, theta=-0), speed=Speed(), color='r', type=type)
    R2 = Robot(ip_addr="192.168.1.102", pos=Position(
        x=-sq2, y=sq2-10/3, theta=0), speed=Speed(), color='g', type=type)
    R3 = Robot(ip_addr="192.168.1.103", pos=Position(
        x=-sq2, y=-sq2-10/3, theta=0), speed=Speed(), color='b', type=type)
    R4 = Robot(ip_addr="192.168.1.104", pos=Position(
        x=sq2, y=-sq2-10/3, theta=0), speed=Speed(), color='y', type=type)

    Robot4 = RobotsSystem(robots=[R1, R2, R3, R4], Neg=6)
    Robot4.history_switch = True

    Robot4.formation_setting(
        init_positions=[Position(x=sq2, y=sq2),
                        Position(x=-sq2, y=sq2),
                        Position(x=-sq2, y=-sq2),
                        Position(x=sq2, y=-sq2)],
        edges=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    filename = "./data/EXPf10s05.xlsx"
    Robot4.add_mission(1, filename)
    Robot4.states_update()
    Robot4.status = "working"
    Robot4.mission_simulation(1)
