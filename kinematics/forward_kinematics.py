'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h25/joints_h25.html
       http://doc.aldebaran.com/2-1/family/nao_h25/links_h25.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for differnt joints
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from keyframes import hello
from numpy.matlib import matrix, identity,sin,cos,dot,array
from angle_interpolation import AngleInterpolationAgent



class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch','LShoulderRoll','LElbowYaw','LElbowRoll'],
                       'RArm': ['RShoulderPitch','RShoulderRoll','RElbowYaw','RElbowRoll'],
                       'LLeg': ['LHipYawPitch','LHipRoll','LHipPitch','LKneePitch','LAnklePitch','LAnkleRoll'],
                       'RLeg': ['RHipYawPitch','RHipRoll','RHipPitch','RKneePitch','RAnklePitch','RAnkleRoll']
                       }

        self.lenJoint = {	'HeadYaw':[0.0,0.0,0.1265],
         					'HeadPitch':[0.0,0.0,0.0],
         					'LShoulderPitch':[0.0,0.098,0.1],
         					'LShoulderRoll':[0.0,0.0,0.0],
         					'LElbowYaw':[0.105,0.015,0.0],
         					'LElbowRoll': [0.0,0.0,0.0],
         					'RShoulderPitch':[0.0,-0.098,0.1],
         					'RShoulderRoll':[0.0,0.0,0.0],
         					'RElbowYaw':[0.105,-0.015,0.0],
         					'RElbowRoll': [0.0,0.0,0.0],
         					'LHipYawPitch':[0.0,0.05,-0.085],
         					'LHipRoll':[0.0,0.0,0.0],
         					'LHipPitch':[0.0,0.0,0.0],
         					'LKneePitch':[0.0,0.0,-0,1],
         					'LAnklePitch':[0.0,0.0,-0.1029],
         					'LAnkleRoll':[0.0,0.0,0.0],
         					'RHipYawPitch':[0.0,-0.05,-0.085],
         					'RHipRoll':[0.0,0.0,0.0],
         					'RHipPitch':[0.0,0.0,0.0],
         					'RKneePitch':[0.0,0.0,-0,1],
         					'RAnklePitch':[0.0,0.0,-0.1029],
         					'RAnkleRoll':[0.0,0.0,0.0]
         				}

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        #T = array((4,4))

        s = sin(joint_angle)
        c = cos(joint_angle)
        
        #x-Rotation
        if joint_name in ['LShoulderRoll','LElbowRoll','RShoulderRoll','RElbowRoll','LHipRoll','LAnkleRoll','RHipRoll','RAnkleRoll']:
            T = array([[1,0,0,0],
                    [0,c,-s,0],
                    [0,s,c,0],
                    [0,0,0,0]])
        
        #y-Rotation
        if joint_name in ['HeadPitch','LShoulderPitch','RShoulderPitch','LHipPitch','RHipPitch','LKneePitch','LAnklePitch','RKneePitch','RAnklePitch']:
            T = array([[c,0,s,0],
                    [0,1,0,0],
                    [-s,0,c,0],
                    [0,0,0,0]])
            
        #z-Rotation
        if joint_name in ['HeadYaw','LElbowYaw','RElbowYaw']: 
            T = array([[c,s,0,0],
                    [-s,c,0,0],
                    [0,0,1,0],
                    [0,0,0,0]])
            
        if joint_name in ['RHipYawPitch','LHipYawPitch']:
            y = array([[c,0,s,0],
                    [0,1,0,0],
                    [-s,0,c,0],
                    [0,0,0,0]])
            
            z = array([[c,s,0,0],
                    [-s,c,0,0],
                    [0,0,1,0],
                    [0,0,0,0]])
            T = y.dot(z)
        
        #print(joint_name)
        #print(T)
  
        
        T[3][0] = self.lenJoint.get(joint_name)[0]
        T[3][1] = self.lenJoint.get(joint_name)[1] 
        T[3][2] = self.lenJoint.get(joint_name)[2]
        T[3][3] = 1

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                 
                T = T.dot(Tl)
                
                self.transforms[joint] = T
             
            for i in self.transforms:
                if i == 'RElbowRoll':
                    x = array(self.transforms.get(i))[3][0]
                    y = array(self.transforms.get(i))[3][1]
                    z = array(self.transforms.get(i))[3][2]
                
            print "x: " , x , " y: ", y , " z: " , z
            

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.keyframes = hello()
    agent.run()
