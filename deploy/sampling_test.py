import numpy as np
from stable_baselines3 import PPO
import argparse
from dynamixel_sdk import *
import time

sys.path.append('./')
import hexapod

# ----------------------- command initialization ----------------------- #

# load command sequence, declare variables

LEN_SEQ = 20
NUM_SAMPLING = 10
command_seq = np.zeros((LEN_SEQ, 18), dtype=np.float32)
sampled_seq = []

# ----------------------- servo initialization ----------------------- #

# set AX 12-A DXL python SDK protocol 1.0 constants

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

# ----------------------- running command sequence ----------------------- #

for line in range(LEN_SEQ):
    last_time = time.time()

    # convert radian into integer ( < 4 ms )

    dxl_goal_pos = list(map(lambda x: int(np.round(x*195.229+511.5)), command_seq[line]))

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

    # sample the joint values

    for s in range(NUM_SAMPLING):

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

        sampled_seq.append(list(map(lambda y: (y-511.5)*5.12218e-3, dxl_present_pos)))

        # wait for dt_action ( dt for sampling : 50 ms )
        # dt_sampling : 1/20 ms, read 20 times, read latency ignored.

        while time.time() - last_time < 0.04995:
            time.sleep(1e-6)

        # printout current info (act, obs)

        # print("goal : ", dxl_goal_pos)
        # print("present : ", dxl_present_pos)
        # print("elapsed time :", time.time() - last_time)
        # print()

f = open("sampled.csv", "w")
for line in range(LEN_SEQ*NUM_SAMPLING):
    for i in range(NUM_DXL):
        f.write(str(sampled_seq[i]))
        if i == NUM_DXL-1:
            f.write("\n")
        else:
            f.write(",")

f.close()
