"""
Basic test script for the Monsoon High Voltage Power Monitor (HVPM / AAA10F),
using USB pass-through mode (phone plugged into the monitor's front USB-A port).

Requires:
    brew install libusb
    pip3 install pymonsoon --break-system-packages

If `pymonsoon` isn't on PyPI for your platform, install directly from source:
    pip3 install git+https://github.com/Monsoon-Solutions-Corporation/pymonsoon.git
"""

import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op

# --- Configuration ---
NUM_SAMPLES = 5000        # ~5000 samples at default sample rate is a few seconds of data
CSV_PATH = "power_readings.csv"

def main():
    # Connect to the monitor over USB (back port, to your Mac)
    mon = HVPM.Monsoon()
    mon.setup_usb()

    # Debug helper: if a method name below throws AttributeError again,
    # uncomment this line to print all real available methods/attributes.
    # print([m for m in dir(mon) if not m.startswith('_')])

    # Set up a sampling engine to record data
    engine = sampleEngine.SampleEngine(mon)

    # We're using USB pass-through (front USB-A to phone), so we want the
    # USB channel's voltage/current, not the Main (banana jack) channel.
    engine.enableChannel(sampleEngine.channels.USBCurrent)
    engine.enableChannel(sampleEngine.channels.USBVoltage)
    engine.disableChannel(sampleEngine.channels.MainCurrent)
    engine.disableChannel(sampleEngine.channels.MainVoltage)

    # Log to CSV so you can graph/analyze later
    engine.enableCSVOutput(CSV_PATH)
    engine.ConsoleOutput(True)

    # Enable USB pass-through mode so power actually flows to the phone
    mon.setUSBPassthroughMode(op.USB_Passthrough.On)

    print(f"Starting capture: {NUM_SAMPLES} samples via USB pass-through...")
    print("Make sure the phone is plugged into the monitor's front USB-A port.")

    # engine.startSampling() streams live output and writes to CSV_PATH.
    # It does not return the data directly (returns None), so we read the CSV back.
    engine.startSampling(NUM_SAMPLES)

    print(f"\nFull dataset saved to {CSV_PATH}")

    # Read back the last row of the CSV as a sanity check
    import csv
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) > 1:
        header = rows[0]
        last_row = rows[-1]
        print("\n--- Last row in CSV ---")
        print(dict(zip(header, last_row)))
    else:
        print("CSV appears empty or only has a header -- check enableCSVOutput() setup.")

    mon.stopSampling()
    mon.closeDevice()

if __name__ == "__main__":
    main()
