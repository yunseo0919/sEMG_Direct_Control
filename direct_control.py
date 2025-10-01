# !/usr/bin/python
# -*- coding:utf-8 -*-

import struct
import time
import asyncio
from collections import deque
import sys

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import matplotlib
matplotlib.use("TkAgg")  # Windows 기본 호환 백엔드
import matplotlib.pyplot as plt

from gforce import DataNotifFlags, GForceProfile, NotifDataType

import numpy as np


# =======================
# Plotting setup
# =======================
CHANNELS = 8
ROWS, COLS = 2, 4               
BUFFER_SAMPLES = 2000                 # 채널별 보존 샘플 수
REFRESH_EVERY_N_PACKETS = 5           # 너무 자주 그리면 느려짐 (패킷 N개마다 업데이트)
YMIN, YMAX = 0, 255                   # 8bit 기본. 12bit면 0~4095로 바꿀 것

channel_buffers = [deque(maxlen=BUFFER_SAMPLES) for _ in range(CHANNELS)]
lines = []
fig = None
axes = None
_plots_initialized = False

# --- Rule-based 동작 분류용 버퍼 ---
WINDOW = 200   # RMS 계산용 윈도우 크기
rms_buffers = [deque(maxlen=WINDOW) for _ in range(CHANNELS)]

# 채널별 동작 매핑
channel_to_action = {
    1: "Wrist Flexion",   # CH1 반응 크면 Wrist Flexion
    3: "Grasp",             #CH3 반응 크면 Grasp
    5: "Wrist Extension",   # CH5 반응 크면 Wrist Extension
    # 필요한 만큼 추가
}

# 채널별 threshold
thresholds = [121.5, 121.5, 121, 121.5, 121, 122, 121, 120]


RMS_WINDOW = 200   # RMS 계산할 샘플 수
rms_texts = []     # 채널별 텍스트 객체 보관


def init_plots():
    global fig, axes, lines, _plots_initialized
    if _plots_initialized:
        return
    plt.ion()
    fig, axes = plt.subplots(ROWS, COLS, figsize=(15, 6), sharex=True)
    axes = axes.flatten()  # 0..7
    fig.suptitle("Real-time EMG")

    lines.clear()
    for ch in range(CHANNELS):
        y = [0] * BUFFER_SAMPLES
        (line,) = axes[ch].plot(y, linewidth=1)
        axes[ch].set_ylabel(f"CH{ch}")
        axes[ch].set_ylim(YMIN, YMAX)     # autoscale 제거 → 딜레이 감소
        lines.append(line)
        # txt = axes[ch].text(0.02, 0.85, "RMS: 0", transform=axes[ch].transAxes)
        # rms_texts.append(txt)

    axes[-2].set_xlabel("samples")
    axes[-1].set_xlabel("samples")
    fig.tight_layout()
    _plots_initialized = True

def update_plots():
    if not _plots_initialized:
        return
    # relim/autoscale 제거 (매 프레임 축 계산 부담 ↓ → 딜레이 감소)
    for ch in range(CHANNELS):
        y = list(channel_buffers[ch])
        if len(y) < BUFFER_SAMPLES:
            y = [0] * (BUFFER_SAMPLES - len(y)) + y
        lines[ch].set_ydata(y)
            # RMS 계산 (최근 RMS_WINDOW 샘플)
        # if len(channel_buffers[ch]) >= RMS_WINDOW:
        #     arr = np.array(list(channel_buffers[ch])[-RMS_WINDOW:], dtype=float)
        #     rms = np.sqrt(np.mean(arr**2))
        # else:
        #     rms = 0
        # rms_texts[ch].set_text(f"RMS: {rms:.1f}")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def classify_action():
    actions = []
    for ch in channel_to_action:
        if len(rms_buffers[ch]) == WINDOW:
            arr = np.array(rms_buffers[ch])
            rms = np.sqrt(np.mean(arr.astype(float) ** 2))
            if rms > thresholds[ch]:
                actions.append(channel_to_action[ch])
    return actions


# =======================
# gForce callbacks
# =======================

def set_cmd_cb(resp):
    print("Command result:", resp)

def get_firmware_version_cb(resp, firmware_version):
    print("Command result:", resp)
    print("Firmware version:", firmware_version)

packet_cnt = 0
start_time = 0
_last_refresh_packet = 0

def ondata(data):
    """
    data[0] : NotifDataType
    data[1:] :
       - EMG raw (len=129): 16 샘플 × 8채널 (8bit 모드), 순차 저장
       - Quaternion (len=17): float 4개
       - Gesture: 가변
    """
    global packet_cnt, start_time, _last_refresh_packet

    if not data:
        return

    dtype = data[0]

    # Quaternion (옵션)
    if dtype == NotifDataType.NTF_QUAT_FLOAT_DATA and len(data) == 17:
        quat_iter = struct.iter_unpack("f", data[1:])
        quaternion = [v[0] for v in quat_iter]
        # print("quaternion:", quaternion)
        return

    # EMG raw (실시간 플롯)
    if dtype == NotifDataType.NTF_EMG_ADC_DATA and len(data) == 129:
        # data_bytes: 128바이트 = 16샘플 * 8채널
        data_bytes = data[1:129]
 
        for i in range(16):
            base = i * CHANNELS
            for ch in range(CHANNELS):
                val = data_bytes[base + ch]
                channel_buffers[ch].append(val) 
                rms_buffers[ch].append(val)      

        # 버퍼 업데이트 끝난 뒤 동작 분류 실행
        actions = classify_action()
        if actions:
            print("Detected:", actions)

                

        # 속도 로그 (100패킷마다)
        if start_time == 0:
            start_time = time.time()
        packet_cnt += 1
        if packet_cnt % 100 == 0:
            period = time.time() - start_time
            sample_rate = 100 * 16 / period  # 16 means repeat times in one packet
            byte_rate = 100 * len(data) / period
            # print(f"----- sample_rate:{sample_rate:.1f}Hz, byte_rate:{byte_rate:.1f}B/s")
            start_time = time.time()

        # 플롯 갱신
        if (packet_cnt - _last_refresh_packet) >= REFRESH_EVERY_N_PACKETS:
            update_plots()
            _last_refresh_packet = packet_cnt

        return

    # Gesture (옵션)
    if dtype == NotifDataType.NTF_EMG_GEST_DATA:
        if len(data) == 2:
            ges = struct.unpack("<B", data[1:])
            print(f"ges_id:{ges[0]}")
        else:
            ges = struct.unpack("<B", data[1:2])[0]
            s = struct.unpack("<H", data[2:4])[0]
            print(f"ges_id:{ges}  strength:{s}")
        return

def print2menu():
    print("_" * 75)
    print("0: Exit")
    print("1: Get Firmware Version")
    print("2: Toggle LED")
    print("3: Toggle Motor")
    print("4: Get Quaternion(press enter to stop)")
    print("5: Set EMG Raw Data Config")
    print("6: Get Raw EMG data(set EMG raw data config first please, press enter to stop)")
    print("7: Get Gesture ID(press enter to stop)")

def wait_key(event):
    input()
    event.set()

async def main():
    # 기본값 (원하면 5번 메뉴에서 바꿀 수 있음)
    sampRate = 500       # 최대 500
    channelMask = 0xFF   # 8채널 모두
    dataLen = 128        # 16*8
    resolution = 8       # 8비트

    event = asyncio.Event()
    gForce = GForceProfile()

    while True:
        print("Scanning devices...")
        scan_results = await gForce.scan(5, "gForce")

        print("_" * 75)
        print("0: exit")

        if not scan_results:
            print("No bracelet was found")
        else:
            for d in scan_results:
                try:
                    print("{0:<1}: {1:^16} {2:<18} Rssi={3:<3}".format(
                        d["index"], d["name"], d["address"], d["rssi"]))
                except Exception as e:
                    print(e)

        try:
            button = int(input("Please select the device you want to connect or exit:"))
        except ValueError:
            button = 0

        if button == 0:
            break

        addr = scan_results[button - 1]["address"]
        await gForce.connect(addr)

        # 플롯 초기화 (연결 후 바로 준비)
        init_plots()

        while True:
            await asyncio.sleep(0.1)
            print2menu()
            try:
                button = int(input("Please select a function or exit:"))
            except ValueError:
                button = 0

            if button == 0:
                break

            elif button == 1:
                await gForce.getControllerFirmwareVersion(get_firmware_version_cb, 1000)

            elif button == 2:
                await gForce.setLED(False, set_cmd_cb, 1000)
                await asyncio.sleep(0.3)
                await gForce.setLED(True, set_cmd_cb, 1000)

            elif button == 3:
                await gForce.setMotor(True, set_cmd_cb, 1000)
                await asyncio.sleep(0.3)
                await gForce.setMotor(False, set_cmd_cb, 1000)

            elif button == 4:
                await gForce.setDataNotifSwitch(DataNotifFlags.DNF_QUATERNION, set_cmd_cb, 1000)
                await asyncio.sleep(0.2)
                await gForce.startDataNotification(ondata)
                print("Press enter to stop...")
                await asyncio.to_thread(wait_key, event)
                await event.wait()
                print("Stopping...")
                await gForce.stopDataNotification()
                await asyncio.sleep(0.2)
                await gForce.setDataNotifSwitch(DataNotifFlags.DNF_OFF, set_cmd_cb, 1000)
                event.clear()

            elif button == 5:
                sampRate = int(input("Please enter sample value(max 500, e.g., 500): "))
                channelMask = eval(input("Please enter channelMask value(e.g., 0xFF): "))
                dataLen = int(input("Please enter dataLen value(e.g., 128): "))
                resolution = int(input("Please enter resolution value(8 or 12, e.g., 8): "))

            elif button == 6:
                await gForce.setEmgRawDataConfig(
                    sampRate, channelMask, dataLen, resolution,
                    cb=set_cmd_cb, timeout=1000
                )
                await gForce.setDataNotifSwitch(DataNotifFlags.DNF_EMG_RAW, set_cmd_cb, 1000)
                await asyncio.sleep(0.2)
                await gForce.startDataNotification(ondata)

                print("Press enter to stop...")
                await asyncio.to_thread(wait_key, event)
                await event.wait()

                print("Stopping...")
                await gForce.stopDataNotification()
                await asyncio.sleep(0.2)
                await gForce.setDataNotifSwitch(DataNotifFlags.DNF_OFF, set_cmd_cb, 1000)
                event.clear()

            elif button == 7:
                flag = int(input("0: gesture only, 1: gesture+strength (0 or 1): "))
                if flag == 0:
                    await gForce.setDataNotifSwitch(DataNotifFlags.DNF_EMG_GESTURE, set_cmd_cb, 1000)
                else:
                    await gForce.setDataNotifSwitch(DataNotifFlags.DNF_EMG_GESTURE_STRENGTH, set_cmd_cb, 1000)
                await asyncio.sleep(0.2)
                await gForce.startDataNotification(ondata)

                print("Press enter to stop...")
                await asyncio.to_thread(wait_key, event)
                await event.wait()

                print("Stopping...")
                await gForce.stopDataNotification()
                await asyncio.sleep(0.2)
                await gForce.setDataNotifSwitch(DataNotifFlags.DNF_OFF, set_cmd_cb, 1000)
                event.clear()

        gForce.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
