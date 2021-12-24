import os
import pybullet
import pybullet_data

def create_joint_position_controller(joint_index=0,lower_limit=-3.14,upper_limit=3.14,inital_position=0):
     # get name of joint, to create on screen label
     joint_info = pybullet.getJointInfo(robot, joint_index)
     #define joint paramters for controller 
     joint_parameters = pybullet.addUserDebugParameter(paramName=str(joint_info[1])+'PC', rangeMin=lower_limit, rangeMax =upper_limit,          startValue=inital_position)
     # return array containing joint index and paramters
     # pass the returned array to activate_position_contoller in the main loop of your script
     return [ joint_index,joint_parameters]

def create_joint_velocity_controller(joint_index=0,lower_limit=-10,upper_limit=10,inital_velosity=0):
  # get name of joint, to create on screen label
   joint_info = pybullet.getJointInfo(robot, joint_index)
    #define joint paramters for controller 
   joint_parameters = pybullet.addUserDebugParameter(paramName=str(joint_info[1])+'VC', rangeMin=lower_limit, rangeMax =upper_limit,        startValue=inital_velosity)
  # return array containing joint index and paramters
   # pass this to activate_position_contoller
   return [ joint_index,joint_parameters]

def activate_position_controller(joint_parameters):
      joint_index = joint_parameters[0] # joint_index th joint position 
      angle = joint_parameters[1]
      user_angle = pybullet.readUserDebugParameter(angle)
      pybullet.setJointMotorControl2(robot, joint_index, pybullet.POSITION_CONTROL,targetPosition= user_angle)
      joint_info = pybullet.getJointState(robot,joint_index)
      joint_position = joint_info[0] 
      joint_velosity = joint_info[1]
      return joint_position,joint_velosity


def activate_velocity_controller(joint_parameters):
      joint_index = joint_parameters[0]
      velosity = joint_parameters[1]
      user_velocity = pybullet.readUserDebugParameter(velosity)
      pybullet.setJointMotorControl2(robot, joint_index, pybullet.VELOCITY_CONTROL,targetVelocity= user_velocity)
      joint_info = pybullet.getJointState(robot,joint_index)
      joint_position = joint_info[0] 
      joint_velosity = joint_info[1]
      return joint_position,joint_velosity

def get_joint_info(robot):
  print('The system has', pybullet.getNumJoints(robot), 'joints')
  num_joints = pybullet.getNumJoints(robot)
  for i in range(num_joints):
      joint_info = pybullet.getJointInfo(robot, i)
      print('Joint number',i)
      print('-------------------------------------')
      print('Joint Index:',joint_info[0])
      print('Joint Name:',joint_info[1])
      print('Joint misc:',joint_info[2:])
      print('-------------------------------------')
  return
  
cid = pybullet.connect(pybullet.DIRECT)
if cid < 0:
    raise ValueError
print("connected")
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
pybullet.resetSimulation()
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81) # define x,y,z gravity constants
pybullet.setTimeStep(0.1)

# Load the URDF file of your robot.

robot = pybullet.loadURDF(r'./hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf',[0,0,0],useFixedBase=1)
#robot = pybullet.loadURDF(r'./pybullet_hexapod_from_simplecar-main/hexapod/resources/ASSY_phantom_urdf/urdf/ASSY_phantom_urdf.urdf',[0,0,0],useFixedBase=1)

#robot = pybullet.loadURDF(r'./Hexy_referenced3/urdf/Hexy_referenced3.urdf',[0,0,0],useFixedBase=1)
get_joint_info(robot)

Joint1_PC = create_joint_position_controller(joint_index =0,lower_limit=-3.14,upper_limit=3.14,inital_position=0)

Joint1_VC = create_joint_velocity_controller(joint_index =1,lower_limit=-10,upper_limit=10,inital_velosity=2)

jointList = [] 
for i in range(18):
 jointList.append(create_joint_position_controller(joint_index =i,lower_limit=-3.14,upper_limit=3.14,inital_position=0.1*i))


def setPos(joint_index,pos):
 _,__=activate_position_controller(create_joint_position_controller(joint_index =joint_index,lower_limit=-3.14,upper_limit=3.14,inital_position=pos))

p = -3.14
cnt = 0
import numpy as np 


################################################ WAY 1 ###################################################
while True:
 pybullet.stepSimulation() 


################################################ WAY 2 ###################################################

'''
while True:

 """ Note you can only activate either a postion or velosity controller, cannot use both simutaneously"""

 #activate Joint1 for on screen control and get position and velosity readings
 cnt += 1 
 p += 0.01 
 print(p)
 
 for i in range(19):
  setPos(i,p)
 
 #joint1_position,joint1_velocity = activate_position_controller(Joint1_PC)

 # comment out the line above and uncomment the below to test the velocity_controller
 
 #joint1_position,joint1_velocity = activate_velocity_controller(Joint1_VC)

 #print(joint1_position,joint1_velocity)

 pybullet.stepSimulation()
pybullet.disconnect()

'''

################################################ WAY 3 ###################################################


'''
while True:

 """ Note you can only activate either a postion or velosity controller, cannot use both simutaneously"""

 #activate Joint1 for on screen control and get position and velosity readings

 joint1_position,joint1_velocity = activate_position_controller(Joint1_PC)

 # comment out the line above and uncomment the below to test the velocity_controller
 
 #joint1_position,joint1_velocity = activate_velocity_controller(Joint1_VC)

 print(joint1_position,joint1_velocity)

 pybullet.stepSimulation()
'''
