from Monsoon import HVPM
from Monsoon import Operations
from Monsoon import sampleEngine
import numpy as np


SAMPLE_COUNT = 10_000  #Sample rate 5000 samples/s


def main():
    monitor = HVPM.Monsoon()

    try:
        print("Connecting to AAA10F...")
        monitor.setup_usb()
        print("Connected.")

        monitor.setUSBPassthroughMode(
            Operations.USB_Passthrough.On
        )

        engine = sampleEngine.SampleEngine(monitor)
        engine.ConsoleOutput(False)

        engine.disableChannel(sampleEngine.channels.MainCurrent)
        engine.disableChannel(sampleEngine.channels.MainVoltage)
        engine.disableChannel(sampleEngine.channels.AuxCurrent)

        engine.enableChannel(sampleEngine.channels.timeStamp)
        engine.enableChannel(sampleEngine.channels.USBCurrent)
        engine.enableChannel(sampleEngine.channels.USBVoltage)

        print("Sampling for about 2 seconds...")
        engine.startSampling(SAMPLE_COUNT)

        samples = engine.getSamples()

        current_ma = np.asarray(
            samples[sampleEngine.channels.USBCurrent],
            dtype=float,
        )
        voltage_v = np.asarray(
            samples[sampleEngine.channels.USBVoltage],
            dtype=float,
        )

        count = min(len(current_ma), len(voltage_v))
        current_ma = current_ma[:count]
        voltage_v = voltage_v[:count]

        if count == 0:
            raise RuntimeError(
                "No USB samples received. Check USB passthrough "
                "and the phone/charger connections."
            )

        power_mw = current_ma * voltage_v

        print()
        print(f"Samples:         {count}")
        print(f"USB voltage:     {voltage_v.mean():.3f} V")
        print(f"USB current:     {current_ma.mean():.3f} mA")
        print(f"USB power:       {power_mw.mean():.3f} mW")
        print(f"USB power:       {power_mw.mean() / 1000:.3f} W")
        print(f"Peak current:    {current_ma.max():.3f} mA")
        print(f"Peak power:      {power_mw.max() / 1000:.3f} W")

    finally:
        try:
            monitor.stopSampling()
        except Exception:
            pass

        try:
            monitor.closeDevice()
        except Exception:
            pass


if __name__ == "__main__":
    main()