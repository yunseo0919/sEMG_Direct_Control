#!/usr/bin/env python3
"""
- 실시간 BLE 데이터 수집
- Bandpass(20–200Hz) + Notch(60Hz) 필터 적용
- RMS(200ms 기본) 계산해서 채널별 플롯
- 이벤트 루프 종료 에러 / shape mismatch 에러 수정
"""

import argparse, asyncio, threading, queue, time
import numpy as np
from collections import deque
from dataclasses import dataclass
from scipy.signal import butter, lfilter, iirnotch

try:
    from bleak import BleakClient, BleakScanner
except Exception:
    BleakClient = BleakScanner = None

# ---------------- 필터 ----------------
def bandpass(sig, fs, low=20, high=200, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return lfilter(b, a, sig)

def notch60(sig, fs, f0=60.0, q=30.0):
    b, a = iirnotch(f0/(fs/2), q)
    return lfilter(b, a, sig)

# ---------------- 스트림 클래스 ----------------
@dataclass
class GForceStream:
    fs:int=500
    n_channels:int=8
    data_bits:int=8                # 8 or 12
    device_name:str="gForce"
    device_address:str=""
    CHAR_UUID:str="f000ffe2-0451-4000-b000-000000000000"
    realtime_pace:bool=True
    _timeout_s:float=10.0

    def __post_init__(self):
        if BleakClient is None:
            raise RuntimeError("bleak 미설치. pip install bleak")
        self.byte_size = 1 if self.data_bits==8 else 2
        self.frame_bytes = self.byte_size * self.n_channels
        self._q:"queue.Queue[np.ndarray]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._last_t = time.time()
        self._rem = bytearray()
        self._got = threading.Event()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._runner, daemon=True).start()
        if not self._got.wait(timeout=self._timeout_s):
            self.close()
            raise RuntimeError("BLE 데이터 없음")

    def __iter__(self): return self
    def __next__(self):
        if self._stop.is_set() and self._q.empty():
            raise StopIteration
        try:
            s = self._q.get(timeout=5.0)
        except queue.Empty:
            raise StopIteration
        if self.realtime_pace:
            target = self._last_t + 1.0/max(1,self.fs)
            dt = target - time.time()
            if dt > 0: time.sleep(dt)
            self._last_t = target
        return s

    def close(self):
        self._stop.set()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except: pass

    def _runner(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ble_main())
        finally:
            # loop.close() 제거 (에러 방지)
            pass

    async def _ble_main(self):
        addr = self.device_address
        if not addr:
            devs = await BleakScanner.discover(timeout=6.0)
            for d in devs:
                if (d.name or "").lower().find(self.device_name.lower())>=0:
                    addr = d.address; break
        if not addr: return

        async with BleakClient(addr) as client:
            if not client.is_connected: return

            def on_notify(_:int, data:bytearray):
                self._rem.extend(data)
                fb=self.frame_bytes
                while len(self._rem)>=fb:
                    frame=self._rem[:fb]; del self._rem[:fb]
                    vals=[]
                    if self.data_bits==8:
                        vals=[float(frame[ch]) for ch in range(self.n_channels)]
                    else:
                        for ch in range(self.n_channels):
                            base=ch*2
                            lo=frame[base]; hi=frame[base+1]
                            raw12=((hi<<8)|lo)&0x0FFF
                            vals.append(float(raw12))
                    arr=np.asarray(vals,dtype=np.float32)
                    try:
                        self._q.put_nowait(arr); self._got.set()
                    except queue.Full:
                        try: _=self._q.get_nowait(); self._q.put_nowait(arr)
                        except: pass

            await client.start_notify(self.CHAR_UUID,on_notify)
            try:
                while not self._stop.is_set():
                    await asyncio.sleep(0.02)
            finally:
                try: await client.stop_notify(self.CHAR_UUID)
                except: pass

# ---------------- RMS Viewer ----------------
def run_rms_view(fs:int, n_channels:int, data_bits:int,
                 address:str, name:str, uuid:str,
                 window_s:float=10, rms_win_ms:int=200):
    import matplotlib.pyplot as plt

    stream=GForceStream(fs=fs,n_channels=n_channels,data_bits=data_bits,
                        device_address=address,device_name=name,
                        CHAR_UUID=uuid,realtime_pace=True)

    win = int(rms_win_ms * fs / 1000)  # RMS 윈도우 샘플 수
    rms_buf=[deque(maxlen=win) for _ in range(n_channels)]
    tdata=[]; rmsdata=[[] for _ in range(n_channels)]
    t0=time.time()

    plt.ion()
    fig, axes = plt.subplots(n_channels,1,figsize=(8,2*n_channels))
    if n_channels==1: axes=[axes]
    raw_buf = [deque(maxlen=fs*2) for _ in range(n_channels)]

    def process_sample(sample):
        """새 샘플 들어올 때마다 RMS 업데이트"""
        rms_vals = []
        for ch in range(n_channels):
            # 1. 채널별 버퍼에 샘플 추가
            raw_buf[ch].append(sample[ch])

            # 2. 버퍼가 충분히 쌓였을 때만 처리
            if len(raw_buf[ch]) >= fs:   # 최소 1초치
                sig = np.array(raw_buf[ch])

                # DC 제거
                sig = sig - np.mean(sig)

                # Bandpass + Notch 필터
                sig = bandpass(sig, fs)
                sig = notch60(sig, fs)

                # RMS (200ms 윈도우)
                win = int(0.2 * fs)
                window = sig[-win:]
                rms_val = np.sqrt(np.mean(window**2))
            else:
                rms_val = 0  # 아직 데이터 부족

            rms_vals.append(rms_val)

        return rms_vals
    lines=[]
    for ch in range(n_channels):
        line,=axes[ch].plot([],[],label=f"CH{ch+1}")
        axes[ch].set_ylim(0,50)   # 필요시 조정
        axes[ch].legend(loc="upper right")
        lines.append(line)

    try:
        while True:
            sample=next(stream)
            for ch in range(n_channels):
                val=sample[ch]
                sig=np.array(raw_buf[ch])
                sig=bandpass([sig],fs)
                sig=notch60(sig,fs)
                rms_buf[ch].append(sig[-1])
                if len(rms_buf[ch])==win:
                    rms_val=np.sqrt(np.mean(np.square(rms_buf[ch])))
                    rmsdata[ch].append(rms_val)
            tdata.append(time.time()-t0)

            # 플롯 업데이트 (조건 추가)
            for ch in range(n_channels):
                if len(rmsdata[ch]) > 0:
                    lines[ch].set_xdata(tdata[-len(rmsdata[ch]):])
                    lines[ch].set_ydata(rmsdata[ch])
                    axes[ch].relim(); axes[ch].autoscale_view()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        stream.close(); plt.ioff(); plt.show()

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser(description="EMG RMS Viewer (exe-style, bugfixed)")
    ap.add_argument("--fs",type=int,default=500)
    ap.add_argument("--channels",type=int,default=8)
    ap.add_argument("--data_bits",type=int,default=8,choices=[8,12])
    ap.add_argument("--device_address",default="")
    ap.add_argument("--device_name",default="gForce")
    ap.add_argument("--uuid",default="f000ffe2-0451-4000-b000-000000000000")
    ap.add_argument("--window",type=float,default=10.0)
    ap.add_argument("--rms_win_ms",type=int,default=200)
    args=ap.parse_args()

    run_rms_view(args.fs,args.channels,args.data_bits,
                 args.device_address,args.device_name,args.uuid,
                 args.window,args.rms_win_ms)

if __name__=="__main__":
    main()
