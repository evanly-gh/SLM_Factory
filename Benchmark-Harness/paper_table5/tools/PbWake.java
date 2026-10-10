// Holds a PARTIAL_WAKE_LOCK as the adb shell user (uid 2000, com.android.shell, which has WAKE_LOCK) so the CPU
// keeps running with the screen off over wireless adb, where no USB connection keeps the phone awake.
// Run with app_process (no app install), the way scrcpy runs its server:
//   CLASSPATH=/data/local/tmp/paper_bench/pbwake.jar app_process / PbWake <stop file>
// Prints "PB_WAKELOCK held", then holds the lock until <stop file> exists (or the process is killed; the binder
// token dies with it and the system releases the lock).

import android.os.Binder;
import android.os.IBinder;

import java.io.File;
import java.lang.reflect.Method;

public class PbWake {
    private static final int PARTIAL_WAKE_LOCK = 1;

    public static void main(String[] args) throws Exception {
        File stop = new File(args.length > 0 ? args[0] : "/data/local/tmp/paper_bench/pbwake.stop");
        Class<?> sm = Class.forName("android.os.ServiceManager");
        IBinder binder = (IBinder) sm.getMethod("getService", String.class).invoke(null, "power");
        Object pm = Class.forName("android.os.IPowerManager$Stub").getMethod("asInterface", IBinder.class)
                .invoke(null, binder);
        Method acquire = null, release = null;
        for (Method m : pm.getClass().getMethods()) {
            if (m.getName().equals("acquireWakeLock")) acquire = m;
            if (m.getName().equals("releaseWakeLock")) release = m;
        }
        if (acquire == null) throw new IllegalStateException("IPowerManager.acquireWakeLock not found");
        IBinder token = new Binder();
        // acquireWakeLock(IBinder lock, int flags, String tag, String packageName, WorkSource ws, String historyTag,
        // int displayId, IWakeLockCallback cb) on recent Android; older versions drop the trailing parameters.
        // Fill by type: ints = flags then displayId (-1), Strings = tag, packageName, historyTag, others null.
        Class<?>[] types = acquire.getParameterTypes();
        Object[] values = new Object[types.length];
        String[] strings = {"pb:benchmark", "com.android.shell", "pb:benchmark"};
        int ints = 0, strs = 0;
        for (int i = 0; i < types.length; i++) {
            if (types[i] == IBinder.class) values[i] = token;
            else if (types[i] == int.class) values[i] = ints++ == 0 ? PARTIAL_WAKE_LOCK : -1;
            else if (types[i] == String.class) values[i] = strs < strings.length ? strings[strs++] : null;
            else if (types[i] == boolean.class) values[i] = false;
            else if (types[i] == long.class) values[i] = 0L;
            else values[i] = null;
        }
        acquire.invoke(pm, values);
        System.out.println("PB_WAKELOCK held");
        System.out.flush();
        while (!stop.exists()) {
            Thread.sleep(1000);
        }
        if (release != null) {
            Object[] rv = new Object[release.getParameterTypes().length];
            rv[0] = token;
            for (int i = 1; i < rv.length; i++) rv[i] = release.getParameterTypes()[i] == int.class ? 0 : null;
            release.invoke(pm, rv);
        }
        System.out.println("PB_WAKELOCK released");
    }
}
