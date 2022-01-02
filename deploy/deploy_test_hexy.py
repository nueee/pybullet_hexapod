import numpy as np
from stable_baselines3 import PPO
import argparse
from dynamixel_sdk import *
import time

sys.path.append('./')
import hexapod

# ----------------------- model initialization ----------------------- #

# load gym env and SB3 model

parser = argparse.ArgumentParser(description="load model, and show how much power is consumed.")
parser.add_argument('--model', required=True, help='zipped model path to measure.')
args = parser.parse_args()

custom_objects = {"learning_rate": lambda x: 3e-5, "lr_schedule": lambda x: 3e-5, "clip_range": lambda x: 0.1}
model = PPO.load(path=str(args.model), custom_objects=custom_objects)

# declare variables

buffer_size = 3
offset1 = np.array([0.0, -0.785398/1.7, 1.362578/1.7]*6)
_jnt_buffer = np.array([offset1]*3, dtype=np.float32)
_act_buffer = np.array([offset1]*3, dtype=np.float32)

# ----------------------- servo initialization ----------------------- #

# set AX 12-A DXL python SDK protocol 1.0 constants

offset = [0, -int(154/1.7), int(267/1.7)]*3+[0, int(154/1.7), -int(267/1.7)]*3

ADDR_AX_TORQUE_ENABLE = 24
ADDR_AX_GOAL_POSITION = 30
ADDR_AX_PRESENT_POSITION = 36

LEN_AX_GOAL_POSITION = 2

PROTOCOL_VERSION = 1.0  # AX-12 A supports protocol 1.0

NUM_DXL = 18
DXL_ID = [1, 3, 5, 13, 15, 17, 7, 9, 11, 2, 4, 6, 14, 16, 18, 8, 10, 12]  # phantomx has 18 servos named ID : 1, 2, ..., 18
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# open the port and set the baudrate

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

groupSyncWrite = GroupSyncWrite(
    port=portHandler,
    ph=packetHandler,
    start_address=ADDR_AX_GOAL_POSITION,
    data_length=LEN_AX_GOAL_POSITION
)

if portHandler.openPort():
    print("opened the port ", DEVICENAME)
else:
    print("port failed")
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("Set the baudrate as ", BAUDRATE)
else:
    print("baudrate failed")
    quit()

# turn on servos

for i in range(NUM_DXL):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        port=portHandler,
        dxl_id=DXL_ID[i],
        address=ADDR_AX_TORQUE_ENABLE,
        data=TORQUE_ENABLE
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error:
        print(packetHandler.getRxPacketError(dxl_error))
    else:
        print("enabled DXL#%02d" % DXL_ID[i])

# declare variables

dxl_goal_pos = [1] * 18
joint_values = [1.0] * 18
dxl_present_pos = [1] * 18
param_goal_pos = [[]] * 18

# reset servos center

for i in range(NUM_DXL):
    param_goal_pos[i] = [
        DXL_LOBYTE(512+offset[i]),
        DXL_HIBYTE(512+offset[i])
    ]
    dxl_addparam_result = groupSyncWrite.addParam(
        dxl_id=DXL_ID[i],
        data=param_goal_pos[i]
    )
    if not dxl_addparam_result:
        print("DXL#%02d groupSyncWrite addparam failed" % DXL_ID[i])
        quit()

dxl_comm_result = groupSyncWrite.txPacket()
if dxl_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(dxl_comm_result))

input("Type any key and enter to start loop.")

# ----------------------- running PPO model ----------------------- #

while True:
    last_time = time.time()

    # get action from observation by model ( < 6 ms )

    observation = np.concatenate([
        _jnt_buffer.ravel(),
        _act_buffer.ravel()
    ])
    action, _ = model.predict(observation.astype(np.float32))
    # Observation : deploy index
    # action : urdf index 

    # convert radian into integer ( < 4 ms )

    dxl_goal_pos[0:7:3] = list(map(lambda x: np.clip(int(np.round(-x*195.229*0.15+511.5)), 0, 1023), action[0:7:3]))
    dxl_goal_pos[1:8:3] = list(map(lambda x: np.clip(int(np.round(x*195.229*0.15+511.5)), 0, 1023), action[1:8:3]))
    dxl_goal_pos[2:9:3] = list(map(lambda x: np.clip(int(np.round(x*195.229*0.15+511.5)), 0, 1023), action[2:9:3]))
    dxl_goal_pos[9:18] = list(map(lambda x: np.clip(int(np.round(-x*195.229*0.15+511.5)), 0, 1023), action[9:18]))
    
    # write action on servos ( < 1 ms )

    for i in range(NUM_DXL):
        param_goal_pos[i] = [
            DXL_LOBYTE(dxl_goal_pos[i]+offset[i]),
            DXL_HIBYTE(dxl_goal_pos[i]+offset[i])
        ]
        dxl_addparam_result = groupSyncWrite.changeParam(
            dxl_id=DXL_ID[i],
            data=param_goal_pos[i]
        )
        if not dxl_addparam_result:
            print("DXL#%02d groupSyncWrite addparam failed" % DXL_ID[i])
            quit()

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
    
    # read state of servos ( ~ 290 ms, but reduce to ~ 35 ms, inexactly )

    for i in range(NUM_DXL):
        dxl_present_pos[i], dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
            port=portHandler,
            dxl_id=DXL_ID[i],
            address=ADDR_AX_PRESENT_POSITION
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error:
            print(packetHandler.getRxPacketError(dxl_error))

    # convert integer into radian ( < 0.01 ms )

    joint_values[0:7:3] = list(map(lambda y: (511.5-y)*5.12218e-3, np.subtract(dxl_present_pos[0:7:3], offset[0:7:3])))
    joint_values[1:8:3] = list(map(lambda y: (y-511.5)*5.12218e-3, np.subtract(dxl_present_pos[1:8:3], offset[1:8:3])))
    joint_values[2:9:3] = list(map(lambda y: (y-511.5)*5.12218e-3, np.subtract(dxl_present_pos[2:9:3], offset[2:9:3])))
    joint_values[9:18] = list(map(lambda y: (511.5-y)*5.12218e-3, np.subtract(dxl_present_pos[9:18], offset[9:18])))
    
    # print(joint_values)

    # update buffer ( < 0.1 ms )

    _jnt_buffer[1:] = _jnt_buffer[:-1]
    _jnt_buffer[0] = joint_values  # get recent joint values
    _act_buffer[1:] = _act_buffer[:-1]
    _act_buffer[0] = action  # get recent action

    # wait for dt_action ( dt for action : 50 ms )

    while time.time() - last_time < 0.04995:
        time.sleep(1e-6)

    # printout current info (act, obs)

    # print("goal : ", dxl_goal_pos)
    # print("present : ", dxl_present_pos)
    # print("elapsed time :", time.time() - last_time)
    # print()

