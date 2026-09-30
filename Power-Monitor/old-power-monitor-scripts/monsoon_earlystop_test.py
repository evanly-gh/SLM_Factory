# monsoon_earlystop_test.py
import time
import threading
from Monsoon import HVPM
from Monsoon import Operations
from Monsoon import sampleEngine
import numpy as np

SAMPLE_COUNT_OVERESTIMATE = 100_000  # way more than we'll actually need

def main():
    monitor = HVPM.Monsoon()
    monitor.setup_usb()
    monitor.setUSBPassthroughMode(Operations.USB_Passthrough.On)

    engine = sampleEngine.SampleEngine(monitor)
    engine.ConsoleOutput(False)
    engine.disableChannel(sampleEngine.channels.MainCurrent)
    engine.disableChannel(sampleEngine.channels.MainVoltage)
    engine.disableChannel(sampleEngine.channels.AuxCurrent)
    engine.enableChannel(sampleEngine.channels.timeStamp)
    engine.enableChannel(sampleEngine.channels.USBCurrent)
    engine.enableChannel(sampleEngine.channels.USBVoltage)

    # Start sampling in a background thread, since startSampling() blocks
    sampling_thread = threading.Thread(
        target=engine.startSampling, args=(SAMPLE_COUNT_OVERESTIMATE,)
    )
    print("Starting sampling in background thread...")
    sampling_thread.start()

    # Simulate "inference happening" for a fixed, known duration
    print("Simulating 3 seconds of 'inference'...")
    time.sleep(3)

    # Try to stop EARLY, before all 100,000 samples are collected
    print("Attempting early stop...")
    monitor.stopSampling()
    sampling_thread.join(timeout=5)

    samples = engine.getSamples()
    current_ma = np.asarray(samples[sampleEngine.channels.USBCurrent], dtype=float)
    voltage_v = np.asarray(samples[sampleEngine.channels.USBVoltage], dtype=float)
    count = min(len(current_ma), len(voltage_v))

    print(f"\nSamples actually captured: {count}")
    print(f"Expected if it ran full duration: {SAMPLE_COUNT_OVERESTIMATE}")
    print(f"Mean current: {current_ma[:count].mean():.3f} mA")
    print(f"Mean voltage: {voltage_v[:count].mean():.3f} V")

    monitor.closeDevice()

if __name__ == "__main__":
    main()
