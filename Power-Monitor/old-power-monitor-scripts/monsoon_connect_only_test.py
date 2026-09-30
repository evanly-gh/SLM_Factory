# monsoon_connect_only_test.py
from Monsoon import HVPM

monitor = HVPM.Monsoon()
print("Attempting to connect...")
monitor.setup_usb()
print("Connected successfully.")
print(f"Serial number: {monitor.statusPacket.serialNumber}")
monitor.closeDevice()
print("Closed cleanly.")
