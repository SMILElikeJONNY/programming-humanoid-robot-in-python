'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello,leftBackToStand,wipe_forehead
import numpy as np
from scipy.misc import comb
from matplotlib import pyplot as plt


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def bernstein(i,n,t):
            return comp(n,i) * (t**(i)) * ((1-t)**(n-i))

    def angle_interpolation(self, keyframes):
        target_joints = {}

        #get the current time in seconds
        currentTime = self.perception.time

        names = np.asarray(self.keyframes[0])
        times = np.asarray(self.keyframes[1])
        keys = self.keyframes[2]


        for joint in range(len(names)):
            i = 0
            for j in range(len(times[joint])):
                if times[joint][j]> currentTime:
                    i = j 
                    break

            #two time values and the two handels
            ### In case the current time is before the first known interpolation time value.
            timeOne = 0
            if i != 0:
                timeOne     = times[joint][i-1]      
            
            timeTwo     = keys[joint][i][1][1]
            timeThree   = keys[joint][i][2][1]
            timeFour    = times[joint][i]

            timeValues  = np.array([timeOne,timeTwo,timeThree,timeFour])

            #Some joints have not a joint
            if self.perception.joint.get(names[joint]) != None:
                #two angles and the handle angles 
                angleOne    = self.perception.joint.get(names[joint])
                angleTwo    = keys[joint][i][1][2]
                angleThree  = keys[joint][i][2][2]
                angleFour   = keys[joint][i][0]

                angleValues = np.array([angleOne,angleTwo,angleThree,angleFour]) 

                #Calc the values
                bernsteinArray = np.array([])

                for i in range(4):
                    x = comb(3,i) * (currentTime**(i)) * ((1- currentTime)**(3-i))
                    bernsteinArray = np.append(bernsteinArray, x)

                #print(angleValues)
                #print(timeValues)

                modTime     = np.dot(bernsteinArray,timeValues)
                modAngle    = np.dot(bernsteinArray,angleValues)

                target_joints[names[joint]] = modAngle


        print(target_joints)
        print("################################")

        return target_joints
       



if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
