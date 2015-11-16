'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
import pickle
from sklearn import svm
import numpy as np


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = pickle.load(open('robot_pose.pkl'))
        
    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        
        current_angle = np.zeros(10)
        
        current_angle[0] = (perception.imu[0])
        current_angle[1] = (perception.imu[1])
        current_angle[2] = (perception.joint.get('LHipYawPitch'))
        current_angle[3] = (perception.joint.get('LHipRoll'))
        current_angle[4] = (perception.joint.get('LHipPitch'))
        current_angle[5] = (perception.joint.get('LKneePitch'))
        current_angle[6] = (perception.joint.get('RHipYawPitch'))
        current_angle[7] = (perception.joint.get('RHipRoll'))
        current_angle[8] = (perception.joint.get('RHipPitch'))
        current_angle[9] = (perception.joint.get('RKneePitch'))
        
        
        
        
        posture =  self.posture_classifier.predict(current_angle.reshape(1,-1))[0]
        
        print(posture)
       
        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
