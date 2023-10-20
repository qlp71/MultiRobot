# 关于MultiRobot的使用方法
## 1.导入包

在```MultiRobot-0.1.tar.gz```所在文件夹下执行
```
pip install MultiRobot-0.1.tar.gz
```
代码中使用
```
# 导入需要的类
from MultiRobot.robot import Robot 
from MultiRobot.robot import RobotSystem 
from MultiRobot.robot import Position 
from MultiRobot.robot import Speed 
```
或者直接用```robot.py```
```
from robot import Robot, RobotSystem, Position, Speed
```

## 2.Robot对象
Robot类的主要属性有

| 属性 | 类型 | 说明 |
| :---: | :---: | :---: |
| ip_addr | str | 机器人ip |
| pos | Position | 机器人位置 |
| speed | Speed | 机器人速度 |
|type|int|机器人类型, 0: 双轮差速,1: 全向移动, 2: carlike|
|name|str|机器人名字(取ip后三位)|

Robot类的主要方法有
| 方法 | 说明 |
| :---: | :---: |
| get_postion | 获取机器人位置 |
|publish_speed|发布机器人速度|
|simu_predict|仿真时预测机器人位置|
|record_history|记录机器人历史轨迹|
|plot_trajectory|绘制机器人历史轨迹|
|plot_control_input|绘制机器人控制输入|

## 3.RobotSystem对象
RobotSystem类的主要属性有

| 属性 | 类型 | 说明 |
| :---: | :---: | :---: |
| robots | list | 机器人列表 |
|edge|list|机器人间的连接关系|
|type|int|机器人类型, 0: 双轮差速,1: 全向移动, 2: carlike|

ps:只有双轮差速机器人通过了实物实验，其他的是仿真实验。

RobotSystem类的主要方法有
| 方法 | 说明 |
| :---: | :---: |
|fromation_settiong|编队设置|
|add_mission|添加任务|
|mission_start|任务开始|
|mission_simulation|任务仿真|

任务仿真和实验都会将机器人的历史轨迹记录在```./data/```文件夹下，excel和json文件。

任务轨迹用excel文件存储路径，如文件```./data/EXPf10s05.xlsx```。

## 4.运动控制方法
懒得解释咋计算的了，后面如果有论文的话看论文吧_(:3」∠)_"
