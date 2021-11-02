#include <string>
#include <sstream>
#include <mutex>
#include <cstring>
#include <nvrtc.h>
#include <thread>


#include "crypto/cn/CryptoNight_monero.h"
#include "CudaCryptonightR_gen.h"
#include "cuda_device.hpp"


static std::string get_code(const V4_Instruction* code, int code_size)
{
    std::stringstream s;

    for (int i = 0; i < code_size; ++i)
    {
        const V4_Instruction inst = code[i];

        const uint32_t a = inst.dst_index;
        const uint32_t b = inst.src_index;

        switch (inst.opcode)
        {
        case MUL:
            s << 'r' << a << "*=r" << b << ';';
            break;

        case ADD:
            s << 'r' << a << "+=r" << b << '+' << inst.C << "U;";
            break;

        case SUB:
            s << 'r' << a << "-=r" << b << ';';
            break;

        case ROR:
            s << 'r' << a << "=rotate_right(r" << a << ",r" << b << ");";
            break;

        case ROL:
            s << 'r' << a << "=rotate_left(r" << a << ",r" << b << ");";
            break;

        case XOR:
            s << 'r' << a << "^=r" << b << ';';
            break;
        }

        s << '\n';
    }

    return s.str();
}

struct CacheEntry
{
    CacheEntry(uint64_t height, int arch_major, int arch_minor, const std::vector<char>& ptx, const std::string& lowered_name) :
        height(height),
        arch_major(arch_major),
        arch_minor(arch_minor),
        ptx(ptx),
        lowered_name(lowered_name)
    {}

    uint64_t height;
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

static std::mutex CryptonightR_cache_mutex;
static std::mutex CryptonightR_build_mutex;
static std::vector<CacheEntry> CryptonightR_cache;

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


static void CryptonightR_build_program(
    std::vector<char>& ptx,
    std::string& lowered_name,
    uint64_t height,
    int arch_major,
    int arch_minor,
    std::string source)
{
    {
        std::lock_guard<std::mutex> g(CryptonightR_cache_mutex);

        // Remove old programs from cache
        for (size_t i = 0; i < CryptonightR_cache.size();) {
            const CacheEntry& entry = CryptonightR_cache[i];
            if (entry.height + 2 < height) {
                CryptonightR_cache[i] = std::move(CryptonightR_cache.back());
                CryptonightR_cache.pop_back();
            }
            else {
                ++i;
            }
        }
    }

    ptx.clear();
    ptx.reserve(65536);

    std::lock_guard<std::mutex> g1(CryptonightR_build_mutex);
    {
        std::lock_guard<std::mutex> g(CryptonightR_cache_mutex);

        // Check if the cache already has this program (some other thread might have added it first)
        for (const CacheEntry& entry : CryptonightR_cache)
        {
            if ((entry.height == height) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor))
            {
                ptx = entry.ptx;
                lowered_name = entry.lowered_name;
                return;
            }
        }
    }

    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(&prog, source.c_str(), "CryptonightR.cu", 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        CUDA_THROW(nvrtcGetErrorString(result));
    }

    result = nvrtcAddNameExpression(prog, "CryptonightR_phase2");
    if (result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);

        CUDA_THROW(nvrtcGetErrorString(result));
    }

    char opt0[64];
    sprintf(opt0, "--gpu-architecture=compute_%d%d", arch_major, arch_minor);

    const char* opts[2] = { opt0, "-DVARIANT=13" };
    result = nvrtcCompileProgram(prog, 2, opts);
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
    result = nvrtcGetLoweredName(prog, "CryptonightR_phase2", &name);
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
        std::lock_guard<std::mutex> g(CryptonightR_cache_mutex);
        CryptonightR_cache.emplace_back(height, arch_major, arch_minor, ptx, lowered_name);
    }
}

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name, uint64_t height, int arch_major, int arch_minor, bool background)
{
    if (background) {
        background_exec([=]() { std::vector<char> tmp; std::string s; CryptonightR_get_program(tmp, s, height, arch_major, arch_minor, false); });
        return;
    }

    ptx.clear();

    const char* source_code_template =
        #include "CryptonightR.cu"
    ;

    const char include_name[] = "XMRIG_INCLUDE_RANDOM_MATH";
    const char *offset = strstr(source_code_template, include_name);
    if (!offset){
        return;
    }

    V4_Instruction code[256];
    const int code_size = v4_random_math_init<xmrig_cuda::Algorithm::CN_R>(code, height);

    std::string source_code(source_code_template, offset);
    source_code.append(get_code(code, code_size));
    source_code.append(offset + sizeof(include_name) - 1);

    {
        std::lock_guard<std::mutex> g(CryptonightR_cache_mutex);

        // Check if the cache has this program
        for (const CacheEntry& entry : CryptonightR_cache) {
            if ((entry.height == height) && (entry.arch_major == arch_major) && (entry.arch_minor == arch_minor)) {
                ptx = entry.ptx;
                lowered_name = entry.lowered_name;

                return;
            }
        }
    }

    CryptonightR_build_program(ptx, lowered_name, height, arch_major, arch_minor, source_code);
}
