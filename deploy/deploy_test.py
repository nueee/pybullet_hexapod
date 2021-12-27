import numpy as np
from stable_baselines3 import PPO
import argparse
from dynamixel_sdk import *
import time

sys.path.append('./')
import hexapod

# ----------------------- SW initialization ----------------------- #

# load gym env and SB3 model

parser = argparse.ArgumentParser(description="load model, and show how much power is consumed.")
parser.add_argument('--model', required=True, help='zipped model path to measure.')
args = parser.parse_args()

custom_objects = {"learning_rate": lambda x: 3e-5, "lr_schedule": lambda x: 3e-5, "clip_range": lambda x: 0.1}
model = PPO.load(path=str(args.model), custom_objects=custom_objects)

# ----------------------- HW initialization ----------------------- #

# set HW constants

ADDR_AX_TORQUE_ENABLE = 24
ADDR_AX_GOAL_POSITION = 30
ADDR_AX_PRESENT_POSITION = 36

LEN_AX_GOAL_POSITION = 2

PROTOCOL_VERSION = 1.0  # AX-12 A supports protocol 1.0

NUM_DXL = 18
DXL_ID = range(1, NUM_DXL+1)  # phantomx has 18 servos named ID : 1, 2, ..., 18
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
dxl_present_pos = [1] * 18
param_goal_pos = [[]] * 18

buffer_size = 3
_jnt_buffer = np.zeros((buffer_size, 18), dtype=np.float32)
_act_buffer = np.zeros((buffer_size, 18), dtype=np.float32)

# reset servos center

for i in range(NUM_DXL):
    param_goal_pos[i] = [
        DXL_LOBYTE(512),
        DXL_HIBYTE(512)
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

    # convert radian into integer ( < 4 ms )

    dxl_goal_pos = list(map(lambda x: np.clip(int(np.round(x*195.229+511.5)), 412, 612), action))

    # write action on servos ( < 1 ms )

    for i in range(NUM_DXL):
        param_goal_pos[i] = [
            DXL_LOBYTE(dxl_goal_pos[i]),
            DXL_HIBYTE(dxl_goal_pos[i])
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
        elif dxl_error != 0:
            print(packetHandler.getRxPacketError(dxl_error))

    # convert integer into radian ( < 0.01 ms )

    joint_values = list(map(lambda y: (y-511.5)*5.12218e-3, dxl_present_pos))

    # update buffer ( < 0.1 ms )

    _jnt_buffer[1:] = _jnt_buffer[:-1]
    _jnt_buffer[0] = joint_values  # get recent joint values
    _act_buffer[1:] = _act_buffer[:-1]
    _act_buffer[0] = action  # get recent action

    # wait for dt_action ( dt for action : 50 ms )

    while time.time() - last_time < 0.0499:
        time.sleep(1e-6)

    # printout current info (act, obs)

    # print("goal : ", dxl_goal_pos)
    # print("present : ", dxl_present_pos)
    # print("elapsed time :", time.time() - last_time)
    # print()

