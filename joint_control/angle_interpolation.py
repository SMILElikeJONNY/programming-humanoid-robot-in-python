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
from keyframes import hello,leftBackToStand,wipe_forehead,leftBellyToStand
import numpy as np
from scipy.misc import comb
from matplotlib import pyplot as plt
import copy
import time
from spark_agent import INVERSED_JOINTS

class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.starttime = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}

        if self.starttime == -1:
            self.starttime = self.perception.time

        # Get the current time in seconds in dependence to the starttime.
        currentTime = self.perception.time - self.starttime
        #print currentTime

        names = self.keyframes[0]
        times = self.keyframes[1]
        keys = self.keyframes[2]


        for joint in range(len(names)):

            # Make sure, that the motion will only execute ones.
            # Joint will be leaved at the last known angle.
            if currentTime < times[joint][len(times[joint])-1]:

                # Get the time slot who discribes between which points we are.
                i = 0   
                for j in range(len(times[joint])):
                    if currentTime < times[joint][j]:
                        i = copy.copy(j) 
                        break


                #two time values and the two handels
                ### In case the current time is before the first known interpolation time value.
                
                timeOne     = 0
                timeTwo     = 0 
                timeThree   = 0
                timeFour    = 0

                angleOne    = 0
                angleTwo    = 0
                angleThree  = 0
                angleFour   = 0

                
                if self.perception.joint.get(names[joint]) != None:
                    #current time before first times point
                    if i == 0:  
                        timeTwo     = times[joint][i] #Point 1
                        timeOne     = keys[joint][i][1][1] + timeTwo #HandleOne before Point 1
                        timeThree   = keys[joint][i][2][1] + timeTwo #HandleTwo after Point 1
                        timeFour    = times[joint][i+1] + keys[joint][i+1][1][1]

                        angleTwo    = keys[joint][i][0]
                        angleOne    = keys[joint][i][1][2] + angleTwo
                        angleThree  = keys[joint][i][2][2] + angleTwo
                        angleFour   = keys[joint][i+1][0] + keys[joint][i+1][1][2]
                    #current time between two time points
                    else:
                        timeOne     = times[joint][i-1]
                        timeTwo     = keys[joint][i-1][2][1] + timeOne
                        timeFour    = times[joint][i]
                        timeThree   = keys[joint][i][1][1] +timeFour

                        angleOne    = keys[joint][i-1][0]
                        angleTwo    = keys[joint][i-1][2][2] + angleOne
                        angleFour   = keys[joint][i][0]
                        angleThree  = keys[joint][i][1][2] + angleFour

                    points = [[timeOne,angleOne],[timeTwo,angleTwo],[timeThree,angleThree],[timeFour,angleFour]]

                    #calculate the angle for every joint
                    t = (currentTime - timeOne)/(timeFour - timeOne)

                    c = np.dot(((1-t)**3),points[0]) + 3*np.dot((((1-t)**2)*t),points[1]) + 3*np.dot(((1-t)*(t**2)),points[2]) + np.dot((t**3),points[3])

                    
                    #some Joints are inversed
                    if names[joint] in INVERSED_JOINTS:
                        target_joints[names[joint]] = c[1]*-1
                    else:
                        target_joints[names[joint]] = c[1]

        return target_joints
       



if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
