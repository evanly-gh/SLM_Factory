# test_monsoon_multi_cycle.py
import time
from monsoon_power import MonsoonPowerMeter

meter = MonsoonPowerMeter()
print("Connected.\n")

for i in range(3):
    print(f"--- Cycle {i+1} ---")
    meter.start_measurement()
    time.sleep(2)  # simulate a short "question"
    result = meter.stop_measurement()
    print(f"  power_ma_mean={result.get('power_ma_mean')} "
          f"sample_count={result.get('sample_count')} "
          f"error={result.get('error')}")
    time.sleep(1)  # brief pause between "questions", like the real pipeline

meter.close()
print("\nClosed cleanly.")
