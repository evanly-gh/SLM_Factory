"""Shared Monsoon HVPM (AAA10F) power-measurement wrapper.

This module exists so that both benchmark pipelines - run_autobench.py
(SmolChat) and run_mnn_autobench.py (MNN Chat) - can measure per-question
power draw off the same Monsoon hardware without duplicating the connection
and sampling logic in two places.

RECOMMENDED API: get_single_reading_via_subprocess(). pymonsoon has been
confirmed, via repeated hardware testing, to work reliably for exactly ONE
measurement per process, then fail at the raw libusb level (USBTimeoutError,
device reset failures) on any subsequent measurement in the SAME process -
regardless of connection strategy (persistent or fresh-per-cycle) or Python
version (3.13/3.14 both showed the identical failure). This rules out
keeping pymonsoon imported in a long-running benchmark process at all.
get_single_reading_via_subprocess() works around this by running
monsoon_single_reading.py as a brand-new subprocess for every single
measurement - pymonsoon is only ever imported once per process, which is
exactly the case that's confirmed to work. Benchmark scripts should call
this per question, not the MonsoonPowerMeter class below.

MonsoonPowerMeter is KEPT FOR REFERENCE/COMPATIBILITY ONLY and is no longer
recommended - see its docstring for why.

Confirmed hardware behavior this module relies on (see monsoon_connect_only_test.py
and monsoon_earlystop_test.py for the raw validation runs):
  - engine.startSampling(N) blocks, so it must run in a background thread.
  - monitor.stopSampling() called from another thread genuinely interrupts
    sampling early; getSamples() then returns whatever was captured up to
    that point, which is exactly what we want for per-question measurements
    inside a much larger overestimated sample budget.
  - The library occasionally logs "Caught disconnection event. Test
    restarting with default parameters" during normal operation. This is
    handled internally by pymonsoon and is not surfaced or treated as an
    error here.

Usage (recommended):
    from monsoon_power import get_single_reading_via_subprocess
    for question in questions:
        power_result = get_single_reading_via_subprocess(duration_seconds=5)
        # merge power_result into that question's metrics

Usage (MonsoonPowerMeter, not recommended - see its docstring):
    meter = MonsoonPowerMeter()          # lightweight, does NOT connect yet
    for question in questions:
        meter.start_measurement()        # fresh connection + starts sampling
        # ... fire broadcast, poll for RUN_DONE ...
        power_result = meter.stop_measurement()  # stops, reads, closes
        # merge power_result into that question's metrics
    meter.close()                        # no-op; kept for compatibility
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

logger = logging.getLogger(__name__)

_SINGLE_READING_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "monsoon_single_reading.py"
)

try:
    from Monsoon import HVPM
    from Monsoon import Operations
    from Monsoon import sampleEngine
    _IMPORT_ERROR = None
except Exception as exc:  # pymonsoon may not be installed on every machine
    HVPM = None
    Operations = None
    sampleEngine = None
    _IMPORT_ERROR = exc

# Observed sustained USB sample rate of the HVPM, used to size the
# generously-overestimated sample budget passed to startSampling().
SAMPLES_PER_SECOND = 5000

# How long to wait, per attempt, for the background sampling thread to exit
# after stopSampling(). stopSampling() is a best-effort signal, not a
# guaranteed-synchronous stop: pymonsoon's internal sampling loop can be mid
# reconnect/retry (the "Caught disconnection event" case) and not honor the
# stop flag until that cycle completes, so a single join is not reliable.
SAMPLING_THREAD_JOIN_TIMEOUT_SECONDS = 10

# How many times to re-signal stopSampling() and re-join before giving up on
# the thread for this measurement.
SAMPLING_THREAD_STOP_MAX_ATTEMPTS = 3

# closeDevice() is itself a USB-level call and has been observed to hang in
# the same way sampling does, so it also runs under a bounded timeout rather
# than being trusted to always return.
CLOSE_DEVICE_TIMEOUT_SECONDS = 10

MONSOON_UNAVAILABLE = "monsoon_unavailable"


def _error_result(error):
    """A result dict with every numeric field set to None, for any failure path."""
    return {
        "power_ma_mean": None,
        "power_ma_std": None,
        "power_ma_max": None,
        "voltage_v_mean": None,
        "power_mw_mean": None,
        "power_mw_max": None,
        "sample_count": 0,
        "source": "monsoon",
        "error": error,
    }


def _safe_close(monitor):
    """Best-effort closeDevice(), tolerant of it hanging or raising.

    closeDevice() is a USB-level call and can hang the same way sampling can,
    so it runs on a background thread under CLOSE_DEVICE_TIMEOUT_SECONDS - a
    hang here must not block the caller forever. Failures are logged, never
    raised, since a failed close must not discard measurement data that was
    already retrieved.
    """
    if monitor is None:
        return

    def _do_close():
        try:
            monitor.closeDevice()
        except Exception as exc:
            logger.warning("closeDevice() raised: %s", exc)

    close_thread = threading.Thread(target=_do_close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=CLOSE_DEVICE_TIMEOUT_SECONDS)
    if close_thread.is_alive():
        logger.warning(
            "closeDevice() did not return within %ds; abandoning it. The "
            "underlying USB handle may not be released, which can cause the "
            "NEXT measurement's fresh connection attempt to fail (reported "
            "as %s).",
            CLOSE_DEVICE_TIMEOUT_SECONDS, MONSOON_UNAVAILABLE,
        )


class MonsoonPowerMeter:
    """Wraps a Monsoon HVPM (AAA10F), reconnecting fresh for every measurement.

    NOT RECOMMENDED ANYMORE - kept for reference/compatibility only. This
    class imports pymonsoon into the calling process and reconnects within
    that same process for every measurement. That has since been confirmed
    unreliable: pymonsoon only works for exactly one measurement per process
    before failing at the libusb level, regardless of whether the connection
    is persistent or, as here, recreated fresh each cycle - the failure is
    per-process, not per-connection. Use get_single_reading_via_subprocess()
    instead, which runs pymonsoon in a fresh subprocess per measurement (the
    one case confirmed to work).

    Construction is lightweight and does not touch the device. Each
    start_measurement()/stop_measurement() pair creates its own brand-new
    HVPM.Monsoon() connection, configures it, samples, and fully closes it
    again - no connection is ever reused across measurements.

    If a fresh connection can't be established, or the sampling thread from
    a measurement won't stop cleanly, start_measurement()/stop_measurement()
    return a "monsoon_unavailable" error result for that one measurement
    instead of raising, so a calling benchmark script can skip Monsoon power
    for that single question (falling back to e.g. BatteryManager's POWER_MA)
    and continue the run rather than aborting.
    """

    def __init__(self):
        self.available = None  # unknown until the first measurement attempt
        self._monitor = None
        self._engine = None
        self._sampling_thread = None
        self._measurement_active = False
        self._start_error = None

    def _connect(self):
        """Create and fully configure a brand-new Monsoon connection.

        Returns (monitor, engine, error) where error is None on success. On
        any failure, releases whatever partial connection was made so we
        don't ourselves contribute to a "device busy" failure on the next
        attempt.
        """
        if HVPM is None:
            return None, None, f"{MONSOON_UNAVAILABLE}: Monsoon library not importable: {_IMPORT_ERROR}"

        monitor = None
        try:
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
            return monitor, engine, None
        except Exception as exc:
            _safe_close(monitor)
            return None, None, f"{MONSOON_UNAVAILABLE}: setup failed: {exc}"

    def _stop_sampling_thread(self):
        """Signal the background sampling thread to stop and verify it actually died.

        stopSampling() is only a signal, not a synchronous stop - the thread
        may still be inside startSampling() (e.g. mid disconnection-retry) by
        the time a single join() times out. This retries stopSampling() +
        join() up to SAMPLING_THREAD_STOP_MAX_ATTEMPTS times, checking
        is_alive() after every attempt, and logs a warning as soon as one
        attempt fails to confirm termination.

        Returns True if the thread is confirmed dead, False if it never
        terminated within the retry budget.
        """
        if self._sampling_thread is None:
            return True

        if not self._sampling_thread.is_alive():
            return True

        for attempt in range(1, SAMPLING_THREAD_STOP_MAX_ATTEMPTS + 1):
            try:
                self._monitor.stopSampling()
            except Exception as exc:
                logger.warning(
                    "stopSampling() raised on attempt %d/%d: %s",
                    attempt, SAMPLING_THREAD_STOP_MAX_ATTEMPTS, exc,
                )

            self._sampling_thread.join(timeout=SAMPLING_THREAD_JOIN_TIMEOUT_SECONDS)

            if not self._sampling_thread.is_alive():
                return True

            logger.warning(
                "Sampling thread still alive %ds after stopSampling() "
                "(attempt %d/%d); retrying stop.",
                SAMPLING_THREAD_JOIN_TIMEOUT_SECONDS, attempt, SAMPLING_THREAD_STOP_MAX_ATTEMPTS,
            )

        logger.warning(
            "Sampling thread did not terminate after %d attempts (%ds total) "
            "for this measurement. Skipping sample retrieval for safety; this "
            "connection will still be torn down and the NEXT measurement "
            "will start a completely fresh one.",
            SAMPLING_THREAD_STOP_MAX_ATTEMPTS,
            SAMPLING_THREAD_STOP_MAX_ATTEMPTS * SAMPLING_THREAD_JOIN_TIMEOUT_SECONDS,
        )
        return False

    def start_measurement(self, overestimate_seconds=30):
        """Create a fresh connection and begin sampling in a background thread.

        The sample count is sized generously for overestimate_seconds at
        SAMPLES_PER_SECOND - the real inference this wraps will always finish
        far sooner, and stop_measurement() interrupts sampling early.

        If a fresh connection can't be established, this records the error
        internally and returns without raising; the error is surfaced when
        stop_measurement() is called, keeping the start/stop call pair
        symmetric for callers.
        """
        self._measurement_active = False
        self._start_error = None

        if self._monitor is not None:
            # A previous measurement was never stopped - tear it down rather
            # than leaking a connection before opening a fresh one.
            _safe_close(self._monitor)
            self._monitor = None
            self._engine = None
            self._sampling_thread = None

        monitor, engine, error = self._connect()
        if error is not None:
            self.available = False
            self._start_error = error
            return

        self._monitor = monitor
        self._engine = engine
        self.available = True

        sample_count = max(1, int(overestimate_seconds * SAMPLES_PER_SECOND))

        try:
            self._sampling_thread = threading.Thread(
                target=self._engine.startSampling, args=(sample_count,), daemon=True
            )
            self._measurement_active = True
            self._sampling_thread.start()
        except Exception as exc:
            self._start_error = f"Failed to start sampling: {exc}"
            self._measurement_active = False
            _safe_close(self._monitor)
            self._monitor = None
            self._engine = None

    def stop_measurement(self):
        """Stop sampling, read back stats, and fully close this measurement's connection.

        Returns a dict with power_ma_mean/std/max, voltage_v_mean,
        power_mw_mean/max, sample_count, and source="monsoon". On any failure
        (fresh connection unavailable, no samples captured, malformed data),
        returns a dict with every numeric field set to None and an "error"
        key describing why, rather than raising.

        The device connection created in start_measurement() is torn down as
        part of this call regardless of outcome - if closeDevice() itself
        hangs or throws, that's caught and logged, but any sample data
        already retrieved is still returned.
        """
        if self._start_error is not None:
            error = self._start_error
            self._start_error = None
            return _error_result(error)

        if not self._measurement_active or self._monitor is None or self._engine is None:
            return _error_result("stop_measurement() called without a matching start_measurement()")

        monitor = self._monitor
        engine = self._engine

        thread_stopped = self._stop_sampling_thread()
        self._measurement_active = False

        if not thread_stopped:
            result = _error_result(
                f"{MONSOON_UNAVAILABLE}: sampling thread failed to terminate for "
                "this measurement; skipping sample retrieval for safety"
            )
        else:
            try:
                samples = engine.getSamples()
                current_ma = np.asarray(samples[sampleEngine.channels.USBCurrent], dtype=float)
                voltage_v = np.asarray(samples[sampleEngine.channels.USBVoltage], dtype=float)

                sample_count = min(len(current_ma), len(voltage_v))
                if sample_count == 0:
                    result = _error_result("No samples captured (sample_count == 0)")
                else:
                    current_ma = current_ma[:sample_count]
                    voltage_v = voltage_v[:sample_count]

                    if not np.all(np.isfinite(current_ma)) or not np.all(np.isfinite(voltage_v)):
                        result = _error_result("Malformed samples (NaN/Inf) in current or voltage readings")
                    else:
                        power_mw = current_ma * voltage_v
                        result = {
                            "power_ma_mean": float(current_ma.mean()),
                            "power_ma_std": float(current_ma.std()),
                            "power_ma_max": float(current_ma.max()),
                            "voltage_v_mean": float(voltage_v.mean()),
                            "power_mw_mean": float(power_mw.mean()),
                            "power_mw_max": float(power_mw.max()),
                            "sample_count": int(sample_count),
                            "source": "monsoon",
                        }
            except Exception as exc:
                result = _error_result(f"Failed to retrieve samples: {exc}")

        # Always tear down this measurement's connection, whatever the
        # outcome above - a fresh connection is created for every
        # measurement, so nothing here is meant to survive to the next one.
        _safe_close(monitor)
        self._monitor = None
        self._engine = None
        self._sampling_thread = None

        return result

    def close(self):
        """Safe to call anytime; kept for backward compatibility.

        Every stop_measurement() already fully closes its own connection, so
        under normal use there is nothing left open here. This only does
        real work if a measurement was started but never stopped.
        """
        if self._measurement_active and self._monitor is not None:
            thread_stopped = self._stop_sampling_thread()
            if not thread_stopped:
                logger.warning(
                    "close() found a sampling thread that would not terminate "
                    "from an incomplete measurement; closeDevice() may race with it."
                )
            _safe_close(self._monitor)

        self._measurement_active = False
        self._start_error = None
        self._monitor = None
        self._engine = None
        self._sampling_thread = None


def get_single_reading_via_subprocess(duration_seconds=5, timeout=30):
    """Take one Monsoon power reading by running monsoon_single_reading.py in a
    fresh subprocess, and return its result dict.

    This is the recommended way to take a Monsoon reading (see module
    docstring for why): pymonsoon is confirmed reliable for exactly one
    measurement per process, so this function gets a fresh process - and
    therefore a fresh pymonsoon import - for every single call, rather than
    keeping pymonsoon loaded in the calling benchmark process across many
    measurements.

    duration_seconds: how long the subprocess should sample for.
    timeout: how long to wait for the subprocess before giving up on it.
        Should comfortably exceed duration_seconds to leave room for process
        startup and device connect/close time.

    Returns the same dict shape monsoon_single_reading.py produces:
    power_ma_mean/std/max, voltage_v_mean, power_mw_mean/max, sample_count,
    source, and error (None on success, a description string otherwise). If
    the subprocess can't be run, times out, or produces output that can't be
    read/parsed, returns an equivalent error dict rather than raising - so a
    calling benchmark script can skip Monsoon power for that one question
    and continue rather than aborting the run.
    """
    output_fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(output_fd)

    try:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    _SINGLE_READING_SCRIPT,
                    "--duration-seconds", str(duration_seconds),
                    "--output-json", output_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _error_result(
                f"{MONSOON_UNAVAILABLE}: monsoon_single_reading.py subprocess "
                f"timed out after {timeout}s"
            )
        except Exception as exc:
            return _error_result(
                f"{MONSOON_UNAVAILABLE}: failed to launch monsoon_single_reading.py "
                f"subprocess: {exc}"
            )

        try:
            with open(output_path, "r") as f:
                content = f.read()
        except Exception as exc:
            return _error_result(
                f"{MONSOON_UNAVAILABLE}: could not read subprocess output file: {exc}; "
                f"stderr: {proc.stderr.strip()}"
            )

        if not content.strip():
            return _error_result(
                f"{MONSOON_UNAVAILABLE}: subprocess produced no output "
                f"(exit code {proc.returncode}); stderr: {proc.stderr.strip()}"
            )

        try:
            return json.loads(content)
        except Exception as exc:
            return _error_result(
                f"{MONSOON_UNAVAILABLE}: could not parse subprocess JSON output: {exc}; "
                f"content: {content[:500]!r}"
            )
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
