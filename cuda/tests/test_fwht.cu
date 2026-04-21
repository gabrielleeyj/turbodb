/**
 * test_fwht.cu — GoogleTest tests for FWHT and core CUDA operations.
 *
 * Tests verify:
 * 1. Context lifecycle (init/destroy).
 * 2. Codebook creation.
 * 3. Rotator creation with correct dimensions.
 * 4. FWHT forward/inverse round-trip preserves vectors.
 * 5. FWHT preserves L2 norms.
 * 6. Determinism: same seed produces same output.
 */

#include <gtest/gtest.h>
#include "turboquant.h"
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

class CUDATest : public ::testing::Test {
protected:
    tq_context_t ctx = nullptr;

    void SetUp() override {
        tq_status_t s = tq_init(0, &ctx);
        if (s != TQ_OK) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    void TearDown() override {
        tq_destroy(ctx);
    }
};

/* --------------------------------------------------------------------------
 * Context tests
 * -------------------------------------------------------------------------- */

TEST_F(CUDATest, ContextLifecycle) {
    ASSERT_NE(ctx, nullptr);

    tq_status_t s = tq_sync(ctx);
    EXPECT_EQ(s, TQ_OK);
}

TEST_F(CUDATest, DeviceInfo) {
    size_t free_bytes, total_bytes;
    EXPECT_EQ(tq_device_memory_info(ctx, &free_bytes, &total_bytes), TQ_OK);
    EXPECT_GT(total_bytes, 0u);
    EXPECT_LE(free_bytes, total_bytes);

    int cc;
    EXPECT_EQ(tq_device_compute_capability(ctx, &cc), TQ_OK);
    EXPECT_GE(cc, 60); /* At least Pascal */
}

/* --------------------------------------------------------------------------
 * Codebook tests
 * -------------------------------------------------------------------------- */

TEST_F(CUDATest, CodebookCreate) {
    /* 2-bit codebook: 4 centroids. */
    float centroids[] = {-0.5f, -0.1f, 0.1f, 0.5f};
    tq_codebook_t cb = nullptr;

    EXPECT_EQ(tq_codebook_create(ctx, centroids, 2, &cb), TQ_OK);
    ASSERT_NE(cb, nullptr);
    EXPECT_EQ(tq_codebook_bit_width(cb), 2);
    EXPECT_EQ(tq_codebook_size(cb), 4);

    tq_codebook_destroy(cb);
}

TEST_F(CUDATest, CodebookInvalidArgs) {
    tq_codebook_t cb = nullptr;
    /* bit_width 0 is invalid. */
    EXPECT_NE(tq_codebook_create(ctx, nullptr, 2, &cb), TQ_OK);
    EXPECT_NE(tq_codebook_create(ctx, (float[]){0.0f}, 0, &cb), TQ_OK);
}

/* --------------------------------------------------------------------------
 * Rotator tests
 * -------------------------------------------------------------------------- */

TEST_F(CUDATest, RotatorCreate) {
    tq_rotator_t rot = nullptr;
    EXPECT_EQ(tq_rotator_create(ctx, 1536, 42, &rot), TQ_OK);
    ASSERT_NE(rot, nullptr);

    EXPECT_EQ(tq_rotator_dim(rot), 1536);
    EXPECT_EQ(tq_rotator_out_dim(rot), 2048); /* next pow2 */

    tq_rotator_destroy(rot);
}

TEST_F(CUDATest, RotatorPow2Dim) {
    tq_rotator_t rot = nullptr;
    EXPECT_EQ(tq_rotator_create(ctx, 256, 99, &rot), TQ_OK);
    EXPECT_EQ(tq_rotator_dim(rot), 256);
    EXPECT_EQ(tq_rotator_out_dim(rot), 256); /* already pow2 */
    tq_rotator_destroy(rot);
}

/* --------------------------------------------------------------------------
 * FWHT tests
 * -------------------------------------------------------------------------- */

TEST_F(CUDATest, FWHTRoundTrip) {
    const int dim = 256;
    const int n = 10;
    const uint64_t seed = 12345;

    tq_rotator_t rot = nullptr;
    ASSERT_EQ(tq_rotator_create(ctx, dim, seed, &rot), TQ_OK);

    int out_dim = tq_rotator_out_dim(rot);

    /* Generate random vectors. */
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(n * out_dim, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            input[i * out_dim + j] = dist(rng);
        }
    }

    /* Upload to device. */
    float *input_d, *forward_d, *inverse_d;
    size_t bytes = n * out_dim * sizeof(float);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&input_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&forward_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&inverse_d), TQ_OK);

    ASSERT_EQ(tq_memcpy_h2d(ctx, input_d, input.data(), bytes), TQ_OK);

    /* Forward FWHT. */
    ASSERT_EQ(tq_fwht_batch(ctx, rot, input_d, n, forward_d), TQ_OK);

    /* Inverse FWHT. */
    ASSERT_EQ(tq_fwht_inverse_batch(ctx, rot, forward_d, n, inverse_d), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    /* Download result. */
    std::vector<float> result(n * out_dim);
    ASSERT_EQ(tq_memcpy_d2h(ctx, result.data(), inverse_d, bytes), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    /* Verify round-trip: original ≈ inverse(forward(original)). */
    for (int i = 0; i < n * out_dim; i++) {
        EXPECT_NEAR(input[i], result[i], 1e-4f)
            << "Mismatch at index " << i;
    }

    tq_device_free(ctx, input_d);
    tq_device_free(ctx, forward_d);
    tq_device_free(ctx, inverse_d);
    tq_rotator_destroy(rot);
}

TEST_F(CUDATest, FWHTPreservesNorm) {
    const int dim = 512;
    const int n = 100;
    const uint64_t seed = 7777;

    tq_rotator_t rot = nullptr;
    ASSERT_EQ(tq_rotator_create(ctx, dim, seed, &rot), TQ_OK);

    int out_dim = tq_rotator_out_dim(rot);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(n * out_dim, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            input[i * out_dim + j] = dist(rng);
        }
    }

    float *input_d, *output_d;
    size_t bytes = n * out_dim * sizeof(float);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&input_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&output_d), TQ_OK);
    ASSERT_EQ(tq_memcpy_h2d(ctx, input_d, input.data(), bytes), TQ_OK);

    ASSERT_EQ(tq_fwht_batch(ctx, rot, input_d, n, output_d), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    std::vector<float> output(n * out_dim);
    ASSERT_EQ(tq_memcpy_d2h(ctx, output.data(), output_d, bytes), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    for (int i = 0; i < n; i++) {
        float norm_in = 0.0f, norm_out = 0.0f;
        for (int j = 0; j < out_dim; j++) {
            norm_in += input[i * out_dim + j] * input[i * out_dim + j];
            norm_out += output[i * out_dim + j] * output[i * out_dim + j];
        }
        norm_in = std::sqrt(norm_in);
        norm_out = std::sqrt(norm_out);

        EXPECT_NEAR(norm_in, norm_out, 1e-3f)
            << "Norm mismatch for vector " << i;
    }

    tq_device_free(ctx, input_d);
    tq_device_free(ctx, output_d);
    tq_rotator_destroy(rot);
}

TEST_F(CUDATest, FWHTDeterminism) {
    const int dim = 128;
    const int n = 5;
    const uint64_t seed = 55555;

    tq_rotator_t rot = nullptr;
    ASSERT_EQ(tq_rotator_create(ctx, dim, seed, &rot), TQ_OK);

    int out_dim = tq_rotator_out_dim(rot);
    size_t bytes = n * out_dim * sizeof(float);

    std::vector<float> input(n * out_dim, 0.0f);
    for (int i = 0; i < n * dim; i++) {
        input[i] = (float)(i % 17) * 0.1f;
    }

    float *input_d, *output1_d, *output2_d;
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&input_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&output1_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, bytes, (void**)&output2_d), TQ_OK);
    ASSERT_EQ(tq_memcpy_h2d(ctx, input_d, input.data(), bytes), TQ_OK);

    /* Run twice. */
    ASSERT_EQ(tq_fwht_batch(ctx, rot, input_d, n, output1_d), TQ_OK);
    ASSERT_EQ(tq_fwht_batch(ctx, rot, input_d, n, output2_d), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    std::vector<float> r1(n * out_dim), r2(n * out_dim);
    ASSERT_EQ(tq_memcpy_d2h(ctx, r1.data(), output1_d, bytes), TQ_OK);
    ASSERT_EQ(tq_memcpy_d2h(ctx, r2.data(), output2_d, bytes), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    for (int i = 0; i < n * out_dim; i++) {
        EXPECT_EQ(r1[i], r2[i]) << "Non-deterministic at index " << i;
    }

    tq_device_free(ctx, input_d);
    tq_device_free(ctx, output1_d);
    tq_device_free(ctx, output2_d);
    tq_rotator_destroy(rot);
}
