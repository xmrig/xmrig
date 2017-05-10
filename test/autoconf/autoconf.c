#include <unity.h>

#include "cpu.h"
#include "options.h"

struct cpu_info cpu_info = { 0 };


static void set_cpu_info(int total_logical_cpus, int l2_cache, int l3_cache) {
    cpu_info.total_cores        = total_logical_cpus;
    cpu_info.total_logical_cpus = total_logical_cpus;
    cpu_info.l2_cache           = l2_cache;
    cpu_info.l3_cache           = l3_cache;
}


void test_autoconf_should_GetOptimalThreadsCounti7(void) {
    set_cpu_info(8, 1024, 8192); // 4C/8T 8 MB (Generic i7 CPU)

    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(2, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(6, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(5, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 60));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 50));
    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 35));
    TEST_ASSERT_EQUAL_INT(2, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 20));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 5));
}


void test_autoconf_should_GetOptimalThreadsCounti5(void) {
    set_cpu_info(4, 1024, 6144); // 2C/4T 6 MB (Generic i5 CPU)

    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 75));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 75));

    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 75));
}


void test_autoconf_should_GetOptimalThreadsCounti3(void) {
    set_cpu_info(4, 512, 3072); // 2C/4T 3 MB (Generic i3 CPU)

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 75));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 75));

    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 75));
}


void test_autoconf_should_GetOptimalThreadsCountR7(void) {
    set_cpu_info(16, 4096, 16384); // 8C/16T 16 MB (AMD Ryzen 7)

    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 75));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 75));

    TEST_ASSERT_EQUAL_INT(16, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 75));
}


void test_autoconf_should_GetOptimalThreadsCountTwoE5620(void) {
    set_cpu_info(16, 2048, 24576); // 8C/16T 24 MB (Two E5620)

    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(6, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 75));
    TEST_ASSERT_EQUAL_INT(6, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 75));

    TEST_ASSERT_EQUAL_INT(16, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(12, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 75));
}


void test_autoconf_should_GetOptimalThreadsCountVCPU(void) {
    set_cpu_info(1, 1024, 15360); // 1C/1T 15 MB (Single core Virtual CPU)

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 75));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 75));

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 75));
}


void test_autoconf_should_GetOptimalThreadsCountNoL3(void) {
    set_cpu_info(8, 8192, 0); // 4C/8T (Multi core Virtual CPU without L3 cache)

    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT, false, 100));
    TEST_ASSERT_EQUAL_INT(2, get_optimal_threads_count(ALGO_CRYPTONIGHT, true, 100));

    TEST_ASSERT_EQUAL_INT(8, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 100));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, true, 100));

    TEST_ASSERT_EQUAL_INT(6, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 75));
    TEST_ASSERT_EQUAL_INT(5, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 60));
    TEST_ASSERT_EQUAL_INT(4, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 50));
    TEST_ASSERT_EQUAL_INT(3, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 35));
    TEST_ASSERT_EQUAL_INT(2, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 20));
    TEST_ASSERT_EQUAL_INT(1, get_optimal_threads_count(ALGO_CRYPTONIGHT_LITE, false, 5));
}


int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_autoconf_should_GetOptimalThreadsCounti7);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCounti5);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCounti3);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCountR7);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCountR7);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCountTwoE5620);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCountVCPU);
    RUN_TEST(test_autoconf_should_GetOptimalThreadsCountNoL3);

    return UNITY_END();
}
