#include <string>
#include <sstream>
#include <mutex>
#include <cstring>
#include <nvrtc.h>
#include <thread>


#include "CudaKawPow_gen.h"
#include "cuda_device.hpp"


struct CacheEntry
{
    CacheEntry(uint64_t period, int arch_major, int arch_minor, const std::vector<char>& ptx, const std::string& lowered_name) :
        period(period),
        arch_major(arch_major),
        arch_minor(arch_minor),
        ptx(ptx),
        lowered_name(lowered_name)
    {}

    uint64_t period;
    int arch_major;
    int arch_minor;
    std::vector<char> ptx;
    std::string lowered_name;
};

struct BackgroundTaskBase
{
    virtual ~BackgroundTaskBase() = default;
    virtual void exec() = 0;
};

template<typename T>
struct BackgroundTask : public BackgroundTaskBase
{
    BackgroundTask(T&& func) : m_func(std::move(func)) {}
    void exec() override { m_func(); }

    T m_func;
};

static std::mutex KawPow_cache_mutex;
static std::mutex KawPow_build_mutex;
static std::vector<CacheEntry> KawPow_cache;

static std::mutex background_tasks_mutex;
static std::vector<BackgroundTaskBase*> background_tasks;
static std::thread* background_thread = nullptr;

static void background_thread_proc()
{
    std::vector<BackgroundTaskBase*> tasks;
    for (;;) {
        tasks.clear();
        {
            std::lock_guard<std::mutex> g(background_tasks_mutex);
            background_tasks.swap(tasks);
        }

        for (BackgroundTaskBase* task : tasks) {
            task->exec();
            delete task;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

template<typename T>
static void background_exec(T&& func)
{
    BackgroundTaskBase* task = new BackgroundTask<T>(std::move(func));

    std::lock_guard<std::mutex> g(background_tasks_mutex);
    background_tasks.push_back(task);
    if (!background_thread) {
        background_thread = new std::thread(background_thread_proc);
    }
}


static inline uint32_t clz(uint32_t a)
{
#ifdef _MSC_VER
    unsigned long index;
    _BitScanReverse(&index, a);
    return 31 - index;
#else
    return __builtin_clz(a);
#endif
}


void calculate_fast_mod_data(uint32_t divisor, uint32_t& reciprocal, uint32_t& increment, uint32_t& shift)
{
    if ((divisor & (divisor - 1)) == 0) {
        reciprocal = 1;
        increment = 0;
        shift = 31U - clz(divisor);
    }
    else {
        shift = 63U - clz(divisor);
        const uint64_t N = 1ULL << shift;
        const uint64_t q = N / divisor;
        const uint64_t r = N - q * divisor;
        if (r * 2 < divisor)
        {
            reciprocal = static_cast<uint32_t>(q);
            increment = 1;
        }
        else
        {
            reciprocal = static_cast<uint32_t>(q + 1);
            increment = 0;
        }
    }
}


static void KawPow_build_program(
    std::vector<char>& ptx,
    std::string& lowered_name,
    uint64_t period,
    int arch_major,
    int arch_minor,
    std::string source)
{
    {
        std::lock_guard<std::mutex> g(KawPow_cache_mutex);

        // Remove old programs from cache
        for (size_t i = 0; i < KawPow_cache.size();) {
            const CacheEntry& entry = KawPow_cache[i];
            if (entry.period + 2 < period) {
                KawPow_cache[i] = std::move(KawPow_cache.back());
                KawPow_cache.pop_back();
            }
            else {
                ++i;
            }
        }
    }

    ptx.clear();
    ptx.reserve(65536);

    std::lock_guard<std::mutex> g1(KawPow_build_mutex);
    {
        std::lock_guard<std::mutex> g(KawPow_cache_mutex);

        // Check if the cache already has this program (some other thread might have added it first)
        for (const CacheEntry& entry : KawPow_cache)
        {
            if ((entry.period == period) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor))
            {
                ptx = entry.ptx;
                lowered_name = entry.lowered_name;
                return;
            }
        }
    }

    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(&prog, source.c_str(), "KawPow.cu", 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        CUDA_THROW(nvrtcGetErrorString(result));
    }

    result = nvrtcAddNameExpression(prog, "progpow_search");
    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }

    char opt0[64];
    sprintf(opt0, "--gpu-architecture=compute_%d%d", arch_major, arch_minor);

    const char* opts[1] = { opt0 };
    result = nvrtcCompileProgram(prog, 1, opts);
    if (result != NVRTC_SUCCESS) {
        size_t logSize;
        if (nvrtcGetProgramLogSize(prog, &logSize) == NVRTC_SUCCESS) {
            char *log = new char[logSize]();
            if (nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
                printf("Program compile log: %s\n", log);
            }

            delete[] log;
        }

        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }


    const char* name;
    result = nvrtcGetLoweredName(prog, "progpow_search", &name);
    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }

    size_t ptxSize;
    result = nvrtcGetPTXSize(prog, &ptxSize);
    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }

    ptx.resize(ptxSize);
    result = nvrtcGetPTX(prog, ptx.data());
    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }

    lowered_name = name;

    nvrtcDestroyProgram(&prog);

    {
        std::lock_guard<std::mutex> g(KawPow_cache_mutex);
        KawPow_cache.emplace_back(period, arch_major, arch_minor, ptx, lowered_name);
    }
}

#define PROGPOW_LANES           16
#define PROGPOW_REGS            32
#define PROGPOW_DAG_LOADS       4
#define PROGPOW_CACHE_BYTES     (16*1024)
#define PROGPOW_CNT_DAG         64
#define PROGPOW_CNT_CACHE       11
#define PROGPOW_CNT_MATH        18

#define rnd()       (kiss99(rnd_state))
#define mix_src()   ("mix[" + std::to_string(rnd() % PROGPOW_REGS) + "]")
#define mix_dst()   ("mix[" + std::to_string(mix_seq_dst[(mix_seq_dst_cnt++) % PROGPOW_REGS]) + "]")
#define mix_cache() ("mix[" + std::to_string(mix_seq_cache[(mix_seq_cache_cnt++) % PROGPOW_REGS]) + "]")

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

static inline uint32_t kiss99(kiss99_t &st)
{
    st.z = 36969 * (st.z & 65535) + (st.z >> 16);
    st.w = 18000 * (st.w & 65535) + (st.w >> 16);
    uint32_t MWC = ((st.z << 16) + st.w);
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    st.jcong = 69069 * st.jcong + 1234567;
    return ((MWC^st.jcong) + st.jsr);
}

static inline uint32_t fnv1a(uint32_t &h, uint32_t d)
{
    return h = (h ^ d) * 0x1000193;
}

// Merge new data from b into the value in a
// Assuming A has high entropy only do ops that retain entropy, even if B is low entropy
// (IE don't do A&B)
static std::string merge(std::string a, std::string b, uint32_t r)
{
    switch (r % 4)
    {
    case 0:
        return a + " = (" + a + " * 33) + " + b + ";\n";
    case 1:
        return a + " = (" + a + " ^ " + b + ") * 33;\n";
    case 2:
        return a + " = ROTL32(" + a + ", " + std::to_string(((r >> 16) % 31) + 1) + ") ^ " + b +
            ";\n";
    case 3:
        return a + " = ROTR32(" + a + ", " + std::to_string(((r >> 16) % 31) + 1) + ") ^ " + b +
            ";\n";
    }
    return "#error\n";
}

// Random math between two input values
static std::string math(std::string d, std::string a, std::string b, uint32_t r)
{
    switch (r % 11)
    {
    case 0:
        return d + " = " + a + " + " + b + ";\n";
    case 1:
        return d + " = " + a + " * " + b + ";\n";
    case 2:
        return d + " = mul_hi(" + a + ", " + b + ");\n";
    case 3:
        return d + " = min(" + a + ", " + b + ");\n";
    case 4:
        return d + " = ROTL32(" + a + ", " + b + " % 32);\n";
    case 5:
        return d + " = ROTR32(" + a + ", " + b + " % 32);\n";
    case 6:
        return d + " = " + a + " & " + b + ";\n";
    case 7:
        return d + " = " + a + " | " + b + ";\n";
    case 8:
        return d + " = " + a + " ^ " + b + ";\n";
    case 9:
        return d + " = clz(" + a + ") + clz(" + b + ");\n";
    case 10:
        return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
    }
    return "#error\n";
}

static void get_code(uint64_t prog_seed, std::string& random_math, std::string& dag_loads)
{
    std::stringstream ret;

    uint32_t seed0 = (uint32_t)prog_seed;
    uint32_t seed1 = prog_seed >> 32;
    uint32_t fnv_hash = 0x811c9dc5;

    kiss99_t rnd_state;
    rnd_state.z = fnv1a(fnv_hash, seed0);
    rnd_state.w = fnv1a(fnv_hash, seed1);
    rnd_state.jsr = fnv1a(fnv_hash, seed0);
    rnd_state.jcong = fnv1a(fnv_hash, seed1);

    // Create a random sequence of mix destinations and cache sources
    // Merge is a read-modify-write, guaranteeing every mix element is modified every loop
    // Guarantee no cache load is duplicated and can be optimized away
    int mix_seq_dst[PROGPOW_REGS];
    int mix_seq_cache[PROGPOW_REGS];
    int mix_seq_dst_cnt = 0;
    int mix_seq_cache_cnt = 0;

    for (int i = 0; i < PROGPOW_REGS; i++)
    {
        mix_seq_dst[i] = i;
        mix_seq_cache[i] = i;
    }

    for (int i = PROGPOW_REGS - 1; i > 0; i--)
    {
        int j;
        j = rnd() % (i + 1);
        std::swap(mix_seq_dst[i], mix_seq_dst[j]);
        j = rnd() % (i + 1);
        std::swap(mix_seq_cache[i], mix_seq_cache[j]);
    }

    for (int i = 0; (i < PROGPOW_CNT_CACHE) || (i < PROGPOW_CNT_MATH); i++)
    {
        if (i < PROGPOW_CNT_CACHE)
        {
            // Cached memory access
            // lanes access random locations
            std::string src = mix_cache();
            std::string dest = mix_dst();
            uint32_t r = rnd();
            ret << "// cache load " << i << "\n";
            ret << "offset = " << src << " % PROGPOW_CACHE_WORDS;\n";
            ret << "data = c_dag[offset];\n";
            ret << merge(dest, "data", r);
        }
        if (i < PROGPOW_CNT_MATH)
        {
            // Random Math
            // Generate 2 unique sources
            int src_rnd = rnd() % ((PROGPOW_REGS - 1) * PROGPOW_REGS);
            int src1 = src_rnd % PROGPOW_REGS; // 0 <= src1 < PROGPOW_REGS
            int src2 = src_rnd / PROGPOW_REGS; // 0 <= src2 < PROGPOW_REGS - 1
            if (src2 >= src1) ++src2; // src2 is now any reg other than src1
            std::string src1_str = "mix[" + std::to_string(src1) + "]";
            std::string src2_str = "mix[" + std::to_string(src2) + "]";
            uint32_t r1 = rnd();
            std::string dest = mix_dst();
            uint32_t r2 = rnd();
            ret << "// random math " << i << "\n";
            ret << math("data", src1_str, src2_str, r1);
            ret << merge(dest, "data", r2);
        }
    }

    random_math = ret.str();

    ret.str(std::string());
    ret << merge("mix[0]", "data_dag.s[0]", rnd());
    for (int i = 1; i < PROGPOW_DAG_LOADS; i++)
    {
        std::string dest = mix_dst();
        uint32_t    r = rnd();
        ret << merge(dest, "data_dag.s[" + std::to_string(i) + "]", r);
    }

    dag_loads = ret.str();
}

void KawPow_get_program(std::vector<char>& ptx, std::string& lowered_name, uint64_t period, uint32_t threads, int arch_major, int arch_minor, const uint64_t* dag_sizes, bool background)
{
    if (background) {
        background_exec([=]() { std::vector<char> tmp; std::string s; KawPow_get_program(tmp, s, period, threads, arch_major, arch_minor, dag_sizes, false); });
        return;
    }

    ptx.clear();

    std::string source_code(
        #include "KawPow.h"
    );

    std::string random_math;
    std::string dag_loads;
    get_code(period, random_math, dag_loads);

    const char random_math_include[] = "XMRIG_INCLUDE_PROGPOW_RANDOM_MATH";
    source_code.replace(source_code.find(random_math_include), sizeof(random_math_include) - 1, random_math);

    const char dag_loads_include[] = "XMRIG_INCLUDE_PROGPOW_DATA_LOADS";
    source_code.replace(source_code.find(dag_loads_include), sizeof(dag_loads_include) - 1, dag_loads);

    constexpr int PERIOD_LENGTH = 3;
    constexpr int EPOCH_LENGTH = 7500;

    const uint64_t epoch = (period * PERIOD_LENGTH) / EPOCH_LENGTH;
    const uint64_t dag_elements = dag_sizes[epoch] / 256;

    uint32_t r, i, s;
    calculate_fast_mod_data(dag_elements, r, i, s);

    std::stringstream ss;
    if (i) {
        ss << "const uint32_t offset1 = offset + " << i << ";\n";
        ss << "const uint32_t rcp = " << r << ";\n";
        ss << "offset -= ((offset1 ? __umulhi(offset1, rcp) : rcp) >> " << (s - 32) << ") * " << dag_elements << ";\n";
    }
    else {
        ss << "offset -= (__umulhi(offset, " << r << ") >> " << (s - 32) << ") * " << dag_elements << ";\n";
    }

    const char offset_mod_include[] = "XMRIG_INCLUDE_OFFSET_MOD_DAG_ELEMENTS";
    source_code.replace(source_code.find(offset_mod_include), sizeof(offset_mod_include) - 1, ss.str());

    ss.str(std::string());

    ss << "__launch_bounds__(" << threads << ", 3)";

    const char launch_bounds_include[] = "XMRIG_INCLUDE_LAUNCH_BOUNDS";
    source_code.replace(source_code.find(launch_bounds_include), sizeof(launch_bounds_include) - 1, ss.str());

    {
        std::lock_guard<std::mutex> g(KawPow_cache_mutex);

        // Check if the cache has this program
        for (const CacheEntry& entry : KawPow_cache) {
            if ((entry.period == period) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor)) {
                ptx = entry.ptx;
                lowered_name = entry.lowered_name;

                return;
            }
        }
    }

    KawPow_build_program(ptx, lowered_name, period, arch_major, arch_minor, source_code);
}
