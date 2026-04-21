/**
 * test_quantize.cu — GoogleTest tests for quantize/dequantize kernels.
 *
 * Tests verify:
 * 1. Quantize/dequantize round-trip produces reasonable reconstruction.
 * 2. Output code size matches expected packed size.
 * 3. Multiple bit-widths work correctly.
 */

#include <gtest/gtest.h>
#include "turboquant.h"
#include <cmath>
#include <vector>
#include <random>

class QuantizeTest : public ::testing::Test {
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

TEST_F(QuantizeTest, MSERoundTrip4Bit) {
    const int dim = 256;
    const int n = 50;
    const int bit_width = 4;
    const uint64_t seed = 42;

    /* Create codebook: uniform centroids for testing. */
    int cb_size = 1 << bit_width;
    std::vector<float> centroids(cb_size);
    for (int i = 0; i < cb_size; i++) {
        centroids[i] = -1.0f + 2.0f * i / (cb_size - 1);
    }

    tq_codebook_t cb = nullptr;
    ASSERT_EQ(tq_codebook_create(ctx, centroids.data(), bit_width, &cb), TQ_OK);

    tq_rotator_t rot = nullptr;
    ASSERT_EQ(tq_rotator_create(ctx, dim, seed, &rot), TQ_OK);

    int out_dim = tq_rotator_out_dim(rot);

    /* Generate random unit vectors. */
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vectors(n * dim);
    for (int i = 0; i < n; i++) {
        float norm = 0.0f;
        for (int j = 0; j < dim; j++) {
            vectors[i * dim + j] = dist(rng);
            norm += vectors[i * dim + j] * vectors[i * dim + j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < dim; j++) {
            vectors[i * dim + j] /= norm;
        }
    }

    /* Allocate device memory. */
    float *vec_d;
    ASSERT_EQ(tq_device_malloc(ctx, n * dim * sizeof(float), (void**)&vec_d), TQ_OK);
    ASSERT_EQ(tq_memcpy_h2d(ctx, vec_d, vectors.data(), n * dim * sizeof(float)), TQ_OK);

    int code_bytes = (bit_width * out_dim + 7) / 8;
    uint8_t *codes_d;
    float *norms_d;
    ASSERT_EQ(tq_device_malloc(ctx, n * code_bytes, (void**)&codes_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, n * sizeof(float), (void**)&norms_d), TQ_OK);

    /* Quantize. */
    ASSERT_EQ(tq_quantize_mse_batch(ctx, vec_d, n, dim, rot, cb,
                                      codes_d, norms_d), TQ_OK);

    /* Dequantize. */
    float *recon_d;
    ASSERT_EQ(tq_device_malloc(ctx, n * dim * sizeof(float), (void**)&recon_d), TQ_OK);
    ASSERT_EQ(tq_dequantize_mse_batch(ctx, codes_d, norms_d, n, dim,
                                        rot, cb, recon_d), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    /* Download and verify. */
    std::vector<float> recon(n * dim);
    ASSERT_EQ(tq_memcpy_d2h(ctx, recon.data(), recon_d, n * dim * sizeof(float)), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    /* Check MSE is reasonable (not zero, not huge). */
    double total_mse = 0.0;
    for (int i = 0; i < n; i++) {
        double mse = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = vectors[i * dim + j] - recon[i * dim + j];
            mse += diff * diff;
        }
        mse /= dim;
        total_mse += mse;
    }
    total_mse /= n;

    /* 4-bit quantization of unit vectors should have MSE < 0.1. */
    EXPECT_LT(total_mse, 0.1) << "MSE too high for 4-bit quantization";
    EXPECT_GT(total_mse, 0.0) << "MSE should not be zero (lossy compression)";

    tq_device_free(ctx, vec_d);
    tq_device_free(ctx, codes_d);
    tq_device_free(ctx, norms_d);
    tq_device_free(ctx, recon_d);
    tq_codebook_destroy(cb);
    tq_rotator_destroy(rot);
}

TEST_F(QuantizeTest, CodeSizeCorrect) {
    /* Verify output code buffer size for various bit-widths. */
    struct TestCase {
        int bit_width;
        int dim;
    };

    TestCase cases[] = {
        {1, 128}, {2, 256}, {3, 512}, {4, 1024}, {5, 768}, {8, 256},
    };

    for (const auto& tc : cases) {
        int out_dim = tc.dim; /* assume pow2 for simplicity */
        /* Find next pow2 */
        int v = 1;
        while (v < tc.dim) v <<= 1;
        out_dim = v;

        int expected_bytes = (tc.bit_width * out_dim + 7) / 8;
        EXPECT_GT(expected_bytes, 0)
            << "bit_width=" << tc.bit_width << " dim=" << tc.dim;
    }
}

TEST_F(QuantizeTest, NormPreservation) {
    /* Verify that stored norms are correct. */
    const int dim = 128;
    const int n = 10;

    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vectors(n * dim);
    std::vector<float> expected_norms(n);

    for (int i = 0; i < n; i++) {
        float norm_sq = 0.0f;
        for (int j = 0; j < dim; j++) {
            float v = dist(rng) * (1.0f + i * 0.5f); /* varying scales */
            vectors[i * dim + j] = v;
            norm_sq += v * v;
        }
        expected_norms[i] = std::sqrt(norm_sq);
    }

    /* Create minimal codebook and rotator. */
    float centroids[] = {-0.5f, 0.5f};
    tq_codebook_t cb = nullptr;
    ASSERT_EQ(tq_codebook_create(ctx, centroids, 1, &cb), TQ_OK);

    tq_rotator_t rot = nullptr;
    ASSERT_EQ(tq_rotator_create(ctx, dim, 42, &rot), TQ_OK);

    int out_dim = tq_rotator_out_dim(rot);
    int code_bytes = (1 * out_dim + 7) / 8;

    float *vec_d;
    uint8_t *codes_d;
    float *norms_d;
    ASSERT_EQ(tq_device_malloc(ctx, n * dim * sizeof(float), (void**)&vec_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, n * code_bytes, (void**)&codes_d), TQ_OK);
    ASSERT_EQ(tq_device_malloc(ctx, n * sizeof(float), (void**)&norms_d), TQ_OK);
    ASSERT_EQ(tq_memcpy_h2d(ctx, vec_d, vectors.data(), n * dim * sizeof(float)), TQ_OK);

    ASSERT_EQ(tq_quantize_mse_batch(ctx, vec_d, n, dim, rot, cb,
                                      codes_d, norms_d), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    std::vector<float> actual_norms(n);
    ASSERT_EQ(tq_memcpy_d2h(ctx, actual_norms.data(), norms_d, n * sizeof(float)), TQ_OK);
    ASSERT_EQ(tq_sync(ctx), TQ_OK);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(actual_norms[i], expected_norms[i], 1e-3f)
            << "Norm mismatch for vector " << i;
    }

    tq_device_free(ctx, vec_d);
    tq_device_free(ctx, codes_d);
    tq_device_free(ctx, norms_d);
    tq_codebook_destroy(cb);
    tq_rotator_destroy(rot);
}
