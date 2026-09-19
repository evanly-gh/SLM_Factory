# test_monsoon_module.py
import time
from monsoon_power import MonsoonPowerMeter

meter = MonsoonPowerMeter()
print("Connected.")

meter.start_measurement()
print("Sampling started, simulating 3 seconds of work...")
time.sleep(3)

result = meter.stop_measurement()
print("Result:", result)

meter.close()
print("Closed cleanly.")
