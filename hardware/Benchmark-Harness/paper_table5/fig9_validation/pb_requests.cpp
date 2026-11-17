// pb_requests: run a list of pre-tokenized requests back to back through MNN's LLM runtime (the same Llm API the
// factory scored with), one PB_MARK line per request on stderr. Built into MNN's tree by build_pb_requests.py.
//
//   pb_requests <config.json> <requests.txt> [--set <json>]... [--ignore-eos 0|1] [--out <ids.jsonl>]
//               [--tune 0|1] [--gap-ms N]
//
// requests.txt: one request per line, "<seq> <warm 0|1> <max_new_tokens> <n> <id_1> ... <id_n>".
//
// Per request (no chat template, no tokenizer on the timed path; the ids are the rendered prompt):
//   begin  = CLOCK_BOOTTIME before generate_init()
//   mid    = after generate(ids, 0): embedding + prefill forward = time to first token (minus one sampling step)
//   end    = after generate(max_new_tokens): the decode loop (sampling + one forward per generated token)
// CLOCK_BOOTTIME is the clock of /proc/uptime, Perfetto and the harness's energy samplers, so energy can be
// integrated over exactly these windows. After each request (outside its timed windows) the process's VmRSS,
// VmHWM, VmSwap and Pss are read from /proc/self.
#include "llm/llm.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace MNN::Transformer;

static unsigned long long pb_boottime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return (unsigned long long) ts.tv_sec * 1000000000ull + (unsigned long long) ts.tv_nsec;
}

// "<key>:   12345 kB" from a /proc file; -1 if absent.
static long proc_kb(const char* path, const char* key) {
    std::ifstream f(path);
    std::string line;
    size_t n = strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, n, key) == 0 && line.size() > n && line[n] == ':') {
            return atol(line.c_str() + n + 1);
        }
    }
    return -1;
}

static long pss_kb() {
    long v = proc_kb("/proc/self/smaps_rollup", "Pss");
    if (v >= 0) return v;
    std::ifstream f("/proc/self/smaps");  // older kernels: sum the per-mapping Pss lines
    std::string line;
    long sum = 0;
    bool any = false;
    while (std::getline(f, line)) {
        if (line.compare(0, 4, "Pss:") == 0) { sum += atol(line.c_str() + 4); any = true; }
    }
    return any ? sum : -1;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o;
}

struct Request {
    int seq = 0, warm = 0, max_new = 0;
    std::vector<int> ids;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <config.json> <requests.txt> [--set <json>]... [--ignore-eos 0|1] [--out <file>] "
                        "[--tune 0|1] [--gap-ms N]\n", argv[0]);
        return 1;
    }
    std::string config_path = argv[1], req_path = argv[2], out_path;
    std::vector<std::string> sets;
    int ignore_eos = 1, tune = 0, gap_ms = 0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "--set") sets.push_back(next());
        else if (a == "--ignore-eos") ignore_eos = atoi(next().c_str());
        else if (a == "--out") out_path = next();
        else if (a == "--tune") tune = atoi(next().c_str());
        else if (a == "--gap-ms") gap_ms = atoi(next().c_str());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    std::vector<Request> reqs;
    {
        std::ifstream f(req_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", req_path.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            Request r;
            int n = 0;
            ss >> r.seq >> r.warm >> r.max_new >> n;
            r.ids.resize(n);
            for (int k = 0; k < n; ++k) ss >> r.ids[k];
            if (!ss || n <= 0) { fprintf(stderr, "bad request line: %.80s\n", line.c_str()); return 1; }
            reqs.push_back(std::move(r));
        }
    }

    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    if (!llm) { fprintf(stderr, "createLLM failed\n"); return 2; }
    // The factory's scoring settings (training.slm_helpers.load_mnn_llm / the HF config.json): greedy, prompt sent
    // verbatim, a fresh context per request. async off so the timings are synchronous (as llm_bench sets it).
    llm->set_config(R"({"async":false,"reuse_kv":false,"sampler_type":"greedy","use_template":false,"tmp_path":"tmp"})");
    for (auto& s : sets) {
        if (!llm->set_config(s)) { fprintf(stderr, "set_config failed: %s\n", s.c_str()); return 2; }
    }
    llm->set_config(std::string("{\"ignore_eos\":") + (ignore_eos ? "true" : "false") + "}");
    unsigned long long l0 = pb_boottime_ns();
    if (!llm->load()) { fprintf(stderr, "load failed\n"); return 3; }
    // as llm_demo and llm_bench do after load(): make the Llm's executor current for this thread
    std::shared_ptr<MNN::Express::Executor::Activation> act = llm->getExecutor()->activte();
    if (tune) {
        llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});  // what llm_bench does (OpenCL encoder batching)
    }
    unsigned long long l1 = pb_boottime_ns();
    fprintf(stderr, "PB_LOAD begin=%llu end=%llu rss_kb=%ld hwm_kb=%ld pss_kb=%ld requests=%d\n", l0, l1,
            proc_kb("/proc/self/status", "VmRSS"), proc_kb("/proc/self/status", "VmHWM"), pss_kb(), (int) reqs.size());
    fprintf(stderr, "PB_CONFIG %s\n", llm->dump_config().c_str());
    fflush(stderr);

    FILE* out = out_path.empty() ? nullptr : fopen(out_path.c_str(), "w");
    const LlmContext* ctx = llm->getContext();
    for (auto& r : reqs) {
        llm->reset();  // as the factory's mnn_generate does before every row: no history carried between requests
        unsigned long long t0 = pb_boottime_ns();
        llm->generate_init(nullptr, nullptr);
        llm->generate(r.ids, 0);  // embedding + prefill; max_new_tokens 0: no decode yet
        unsigned long long t1 = pb_boottime_ns();
        long long prefill_us = ctx->prefill_us;
        int prompt_len = ctx->prompt_len;
        int status_prefill = (int) ctx->status;
        if (r.max_new > 0 && ctx->status != LlmStatus::INTERNAL_ERROR) {
            llm->generate(r.max_new);
        }
        unsigned long long t2 = pb_boottime_ns();
        int gen = ctx->gen_seq_len;
        int status = (int) ctx->status;
        // Decode forward passes: each generated token costs one forward, except an end-of-turn token, which stops
        // the loop before its forward (status NORMAL_FINISHED).
        int fwd = (ctx->status == LlmStatus::NORMAL_FINISHED && gen > 0) ? gen - 1 : gen;
        long rss = proc_kb("/proc/self/status", "VmRSS"), hwm = proc_kb("/proc/self/status", "VmHWM");
        long swp = proc_kb("/proc/self/status", "VmSwap"), pss = pss_kb();
        fprintf(stderr, "PB_MARK req seq=%d warm=%d begin=%llu mid=%llu end=%llu prompt=%d gen=%d fwd=%d max_new=%d "
                        "prefill_us=%lld decode_us=%lld status_prefill=%d status=%d rss_kb=%ld hwm_kb=%ld swap_kb=%ld "
                        "pss_kb=%ld\n",
                r.seq, r.warm, t0, t1, t2, prompt_len, gen, fwd, r.max_new, prefill_us, (long long) ctx->decode_us,
                status_prefill, status, rss, hwm, swp, pss);
        fflush(stderr);
        if (out) {
            fprintf(out, "{\"seq\":%d,\"warm\":%d,\"ids\":[", r.seq, r.warm);
            for (size_t k = 0; k < ctx->output_tokens.size(); ++k) {
                fprintf(out, k ? ",%d" : "%d", ctx->output_tokens[k]);
            }
            fprintf(out, "],\"text\":\"%s\"}\n", json_escape(ctx->generate_str).c_str());
            fflush(out);
        }
        if (status == (int) LlmStatus::INTERNAL_ERROR) {
            fprintf(stderr, "request %d: internal error, stopping\n", r.seq);
            if (out) fclose(out);
            return 4;
        }
        if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
    }
    if (out) fclose(out);
    fprintf(stderr, "PB_DONE hwm_kb=%ld\n", proc_kb("/proc/self/status", "VmHWM"));
    return 0;
}
