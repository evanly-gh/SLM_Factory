import argparse
import json
from Monsoon import HVPM, Operations, sampleEngine
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=5)
    args = parser.parse_args()

    result = {"power_ma_mean": None, "sample_count": 0, "error": None}
    monitor = None
    try:
        monitor = HVPM.Monsoon()
        monitor.setup_usb()

        # Device stays connected while performing monitoring
        monitor.setUSBPassthroughMode(Operations.USB_Passthrough.On) 

        # Set up engine to only sample timestamp, current, and voltage
        # Disable other channels as we only care about USB
        engine = sampleEngine.SampleEngine(monitor)
        engine.ConsoleOutput(False)
        engine.disableChannel(sampleEngine.channels.MainCurrent)
        engine.disableChannel(sampleEngine.channels.MainVoltage)
        engine.disableChannel(sampleEngine.channels.AuxCurrent)
        engine.enableChannel(sampleEngine.channels.timeStamp)
        engine.enableChannel(sampleEngine.channels.USBCurrent)
        engine.enableChannel(sampleEngine.channels.USBVoltage)

        # Start sampling
        engine.startSampling(int(args.duration_seconds * 5000))
        samples = engine.getSamples()
        current = np.asarray(samples[sampleEngine.channels.USBCurrent], dtype=float)

        # "Power" (in terms of mA) is averaged over samples and becomes the result
        if len(current) > 0:
            result["power_ma_mean"] = float(current.mean())
            result["sample_count"] = len(current)
        else:
            result["error"] = "No USB samples received"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    finally:
        if monitor is not None:
            try:
                monitor.stopSampling()
            except Exception:
                pass
            try:
                monitor.closeDevice()
            except Exception:
                pass

    print(json.dumps(result))


if __name__ == "__main__":
    main()