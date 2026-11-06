// Battery current / charge sampler through Android's battery service (BatteryManager.getLongProperty, the same
// health-HAL values an app reads without permissions), for phones whose /sys/class/power_supply files are hidden
// from the shell user and whose Perfetto cannot record the battery counters (OnePlus 8 Pro, Android 13).
// Run with app_process as the shell user (no app install):
//   CLASSPATH=/data/local/tmp/paper_bench/pbbattery.jar app_process / PbBattery <out file> <stop file> [interval ms]
//   CLASSPATH=... app_process / PbBattery --once      (prints one sample, for discovery)
// Writes the energy sampler's format: a "# up <columns>" header, then "<CLOCK_BOOTTIME s> <values>" rows, so the
// rows line up with PB_MARK timestamps. Voltage is not a battery property; it comes from `dumpsys battery` (mV)
// every 5 s and is held in between (it changes slowly). Unavailable values are written as "NA".

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PbBattery {
    private static final int CHARGE_COUNTER = 1, CURRENT_NOW = 2, CURRENT_AVERAGE = 3;
    private static final Pattern VOLTAGE = Pattern.compile("^\\s*voltage:\\s*(\\d+)", Pattern.MULTILINE);

    private static String uptime() throws Exception {
        try (BufferedReader r = new BufferedReader(new FileReader("/proc/uptime"))) {
            return r.readLine().split(" ")[0];
        }
    }

    private static String voltageMv() {
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"dumpsys", "battery"});
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                for (String line; (line = r.readLine()) != null; ) sb.append(line).append('\n');
            }
            p.waitFor();
            Matcher m = VOLTAGE.matcher(sb);
            return m.find() ? m.group(1) : "NA";
        } catch (Exception e) {
            return "NA";
        }
    }

    private static String prop(Object bm, Method get, int id) throws Exception {
        long v = (Long) get.invoke(bm, id);
        return v == Long.MIN_VALUE || v == Integer.MIN_VALUE ? "NA" : Long.toString(v);
    }

    public static void main(String[] args) throws Exception {
        Object bm = Class.forName("android.os.BatteryManager").getDeclaredConstructor().newInstance();
        Method get = bm.getClass().getMethod("getLongProperty", int.class);
        String header = "# up bm:current_now bm:current_avg bm:charge_counter bm:voltage_now";
        if (args.length > 0 && args[0].equals("--once")) {
            System.out.println(header);
            System.out.println(uptime() + " " + prop(bm, get, CURRENT_NOW) + " " + prop(bm, get, CURRENT_AVERAGE)
                    + " " + prop(bm, get, CHARGE_COUNTER) + " " + voltageMv());
            return;
        }
        File stop = new File(args[1]);
        long interval = args.length > 2 ? Long.parseLong(args[2]) : 100;
        String volt = voltageMv();
        long lastVolt = System.nanoTime();
        try (PrintWriter out = new PrintWriter(new FileWriter(args[0]))) {
            out.println(header);
            System.out.println("PB_BATTERY sampling");
            System.out.flush();
            while (!stop.exists()) {
                if (System.nanoTime() - lastVolt > 5_000_000_000L) {
                    volt = voltageMv();
                    lastVolt = System.nanoTime();
                }
                out.println(uptime() + " " + prop(bm, get, CURRENT_NOW) + " " + prop(bm, get, CURRENT_AVERAGE)
                        + " " + prop(bm, get, CHARGE_COUNTER) + " " + volt);
                out.flush();
                Thread.sleep(interval);
            }
        }
    }
}
