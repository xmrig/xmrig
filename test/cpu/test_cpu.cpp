#include <unity.h>
#include <libcpuid.h>
#include <iostream>

#include "Options.h"
#include "Cpu.h"

struct cpu_id_t mockCpuId;

int cpuid_get_raw_data(struct cpu_raw_data_t* data)
{
    return 0;
}

int cpu_identify(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
    memcpy(data, &mockCpuId, sizeof(struct cpu_id_t));
    return 0;
}

void setMockedCpu(size_t numProcessors, size_t numCores, size_t numPusPerCore, size_t l3Cache)
{
    strcpy(mockCpuId.brand_str, "CPU Test Brand");
    mockCpuId.vendor = VENDOR_INTEL;

    mockCpuId.num_cores = numCores;
    mockCpuId.num_logical_cpus = numCores * numPusPerCore;
    mockCpuId.total_logical_cpus = mockCpuId.num_logical_cpus * numProcessors;
    mockCpuId.l3_cache = l3Cache;
    mockCpuId.l2_cache = 128;

    Cpu::init();
}

std::pair<size_t, size_t> testOptimize(size_t numThreads, size_t hashFactor, Options::Algo algo, bool safeMode,
                                       size_t maxCpuUsage = 100)
{
    Cpu::optimizeParameters(numThreads, hashFactor, algo, maxCpuUsage, safeMode);
    return std::pair<size_t, size_t>(numThreads, hashFactor);
}

class Expected
{
public:
    typedef std::pair<size_t, size_t> value_type;

public:
    Expected(size_t threadCount, size_t hashFactor) :
            m_expectedValues(threadCount,
                             std::min(hashFactor,
                                      static_cast<size_t>(MAX_NUM_HASH_BLOCKS)))
    {
    }

    bool operator==(const value_type& actualValues)
    {
        if (m_expectedValues != actualValues)
        {
            std::cout << "Mismatch:"
                      << " expected=(" << m_expectedValues.first << "," << m_expectedValues.second <<")"
                      << " actual=(" << actualValues.first << "," << actualValues.second << ")" << std::endl;

        }
        return m_expectedValues == actualValues;
    }

private:
    value_type m_expectedValues;
};

void test_cpu_optimizeparameters_p1_c1_v1_m1(void)
{
    const size_t NUM_PROCESSORS = 1;
    const size_t NUM_CORES = 1;
    const size_t NUM_PUS_PER_CORE = 1;
    const size_t L3_CACHE = 1024;
    setMockedCpu(NUM_PROCESSORS, NUM_CORES, NUM_PUS_PER_CORE, L3_CACHE);

    TEST_ASSERT_EQUAL_UINT32(Cpu::availableCache(), L3_CACHE);

    TEST_ASSERT(Expected(1,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, true));
}

void test_cpu_optimizeparameters_p1_c1_v2_m2(void)
{
    const size_t NUM_PROCESSORS = 1;
    const size_t NUM_CORES = 1;
    const size_t NUM_PUS_PER_CORE = 2;
    const size_t L3_CACHE = 2048;
    setMockedCpu(NUM_PROCESSORS, NUM_CORES, NUM_PUS_PER_CORE, L3_CACHE);

    TEST_ASSERT_EQUAL_UINT32(Cpu::availableCache(), L3_CACHE);

    TEST_ASSERT(Expected(1,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,2) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(2,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,2) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(2,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, true));
}

void test_cpu_optimizeparameters_p1_c4_v2_m8(void)
{
    const size_t NUM_PROCESSORS = 1;
    const size_t NUM_CORES = 4;
    const size_t NUM_PUS_PER_CORE = 2;
    const size_t L3_CACHE = 8 * 1024;
    setMockedCpu(NUM_PROCESSORS, NUM_CORES, NUM_PUS_PER_CORE, L3_CACHE);

    TEST_ASSERT_EQUAL_UINT32(Cpu::availableCache(), L3_CACHE);

    TEST_ASSERT(Expected(4,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(4,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 80));
    TEST_ASSERT(Expected(3,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 48));
    TEST_ASSERT(Expected(3,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 38));
    TEST_ASSERT(Expected(2,2) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 37));
    TEST_ASSERT(Expected(2,2) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 25));
    TEST_ASSERT(Expected(1,4) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 24));
    TEST_ASSERT(Expected(1,4) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 1));
    TEST_ASSERT(Expected(1,4) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false, 0));
    TEST_ASSERT(Expected(8,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,4) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,8) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(2,2) == testOptimize(2, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,4) == testOptimize(2, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(3,1) == testOptimize(3, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(3,2) == testOptimize(3, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(4,1) == testOptimize(4, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(4,2) == testOptimize(4, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(4,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(2,2) == testOptimize(0, 2, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(4,2) == testOptimize(0, 2, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,3) == testOptimize(0, 3, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,3) == testOptimize(0, 3, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,4) == testOptimize(0, 4, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,4) == testOptimize(0, 4, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(4,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(8,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,4) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,8) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(4,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(8,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, true));
}

void test_cpu_optimizeparameters_p1_c8_v1_m25(void)
{
    const size_t NUM_PROCESSORS = 1;
    const size_t NUM_CORES = 8;
    const size_t NUM_PUS_PER_CORE = 1;
    const size_t L3_CACHE = 25 * 1024;
    setMockedCpu(NUM_PROCESSORS, NUM_CORES, NUM_PUS_PER_CORE, L3_CACHE);

    TEST_ASSERT_EQUAL_UINT32(Cpu::availableCache(), L3_CACHE);

    TEST_ASSERT(Expected(8,1) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,3) == testOptimize(0, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,12) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,25) == testOptimize(1, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(2,6) == testOptimize(2, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(2,12) == testOptimize(2, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(3,4) == testOptimize(3, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(3,8) == testOptimize(3, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(4,3) == testOptimize(4, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(4,6) == testOptimize(4, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(5,2) == testOptimize(5, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(5,5) == testOptimize(5, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(6,2) == testOptimize(6, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(6,4) == testOptimize(6, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(7,1) == testOptimize(7, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(7,3) == testOptimize(7, 0, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(8,1) == testOptimize(8, 0, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,3) == testOptimize(8, 0, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(8,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,1) == testOptimize(0, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(6,2) == testOptimize(0, 2, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,2) == testOptimize(0, 2, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(4,3) == testOptimize(0, 3, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(8,3) == testOptimize(0, 3, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(3,4) == testOptimize(0, 4, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(6,4) == testOptimize(0, 4, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(2,5) == testOptimize(0, 5, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(5,5) == testOptimize(0, 5, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(2,6) == testOptimize(0, 6, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(4,6) == testOptimize(0, 6, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,7) == testOptimize(0, 7, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(3,7) == testOptimize(0, 7, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,8) == testOptimize(0, 8, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(3,8) == testOptimize(0, 8, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,1) == testOptimize(1, 1, Options::ALGO_CRYPTONIGHT_LITE, false));

    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(8,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(8,1) == testOptimize(10, 1, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(1,10) == testOptimize(1, 10, Options::ALGO_CRYPTONIGHT_LITE, true));

    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, false));
    TEST_ASSERT(Expected(10,10) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, false));
    TEST_ASSERT(Expected(8,1) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT, true));
    TEST_ASSERT(Expected(8,3) == testOptimize(10, 10, Options::ALGO_CRYPTONIGHT_LITE, true));
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_cpu_optimizeparameters_p1_c1_v1_m1);
    RUN_TEST(test_cpu_optimizeparameters_p1_c1_v2_m2);
    RUN_TEST(test_cpu_optimizeparameters_p1_c4_v2_m8);
    RUN_TEST(test_cpu_optimizeparameters_p1_c8_v1_m25);

    return UNITY_END();
}
