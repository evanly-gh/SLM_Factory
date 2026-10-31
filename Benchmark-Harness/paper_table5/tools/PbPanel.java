// Turns the display PANEL off while Android stays interactive ("Awake"), the way scrcpy's --turn-screen-off does.
// Some OEM builds lower CPU frequency caps as soon as the phone goes to sleep (OnePlus 8 Pro / OxygenOS 13: the
// policy4 cluster drops from 2.42 to 1.48 GHz with the screen asleep), so a real screen-off run is slower than the
// same run with the screen on. With the panel off but the system awake, those policies do not trigger and the
// display draws (almost) no power.
// Run with app_process as the shell user (no app install):
//   CLASSPATH=/data/local/tmp/paper_bench/pbpanel.jar app_process / PbPanel <stop file>
// Prints "PB_PANEL off", re-applies the power mode every 2 s (Android may restore it on a display state change),
// and turns the panel back on (POWER_MODE_NORMAL) when <stop file> appears. Android 10-13 (SurfaceControl API).
// Every 60 s it also injects a Ctrl key press (user activity): OxygenOS caps the screen timeout at 30 min whatever
// screen_off_timeout says, and an asleep phone gets the CPU cap back (2026-10-20: asleep after 30 min of a run).

import android.os.IBinder;

import java.io.File;
import java.lang.reflect.Method;

public class PbPanel {
    private static final int POWER_MODE_OFF = 0;
    private static final int POWER_MODE_NORMAL = 2;

    private static IBinder displayToken(Class<?> sc) throws Exception {
        try {
            return (IBinder) sc.getMethod("getInternalDisplayToken").invoke(null);
        } catch (NoSuchMethodException e) {
            long[] ids = (long[]) sc.getMethod("getPhysicalDisplayIds").invoke(null);
            return (IBinder) sc.getMethod("getPhysicalDisplayToken", long.class).invoke(null, ids[0]);
        }
    }

    public static void main(String[] args) throws Exception {
        File stop = new File(args.length > 0 ? args[0] : "/data/local/tmp/paper_bench/pbpanel.stop");
        // "keep": keep the phone awake but leave the panel on (screen on at minimum brightness). llama.cpp's OpenCL
        // backend with Q8_0/F16 weights kernel-panicked the OnePlus 8 Pro three times with the panel off (2026-10-21/22)
        boolean panelOff = !(args.length > 1 && args[1].equals("keep"));
        Class<?> sc = Class.forName("android.view.SurfaceControl");
        Method setMode = sc.getMethod("setDisplayPowerMode", IBinder.class, int.class);
        IBinder token = displayToken(sc);
        if (token == null) throw new IllegalStateException("no built-in display token");
        if (panelOff) setMode.invoke(null, token, POWER_MODE_OFF);
        System.out.println(panelOff ? "PB_PANEL off" : "PB_PANEL keep-awake");
        System.out.flush();
        Object pm = Class.forName("android.os.IPowerManager$Stub").getMethod("asInterface", IBinder.class).invoke(null,
                Class.forName("android.os.ServiceManager").getMethod("getService", String.class).invoke(null, "power"));
        Method interactive = pm.getClass().getMethod("isInteractive");
        long lastPoke = 0;
        while (!stop.exists()) {
            // An asleep OnePlus 8 Pro caps policy4 at 1.48 GHz within seconds (SmolLM2-135M Q8_0 CPU prefill 124-228
            // instead of 918 tokens/s), so check every 2 s and wake it at once, not only at the 60 s poke.
            boolean asleep = !(Boolean) interactive.invoke(pm);
            if (asleep || System.currentTimeMillis() - lastPoke > 60_000) {
                // KEYCODE_WAKEUP wakes a phone that fell asleep anyway; KEYCODE_CTRL_LEFT is dispatched to the focused
                // window, which counts as user activity and restarts the screen timeout
                Runtime.getRuntime().exec(new String[]{"sh", "-c", "input keyevent 224; input keyevent 113"}).waitFor();
                lastPoke = System.currentTimeMillis();
            }
            Thread.sleep(2000);
            if (panelOff) setMode.invoke(null, token, POWER_MODE_OFF);
        }
        if (panelOff) setMode.invoke(null, token, POWER_MODE_NORMAL);
        System.out.println("PB_PANEL on");
    }
}
