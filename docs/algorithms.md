# TurboQuant Algorithm Reference

Reference: arXiv:2504.19874v1

## Core Insight

TurboQuant is a data-oblivious (zero-calibration) vector quantization scheme.
It uses random rotation followed by scalar quantization to achieve near-optimal
distortion without needing to see the data distribution.

## Algorithm 1: QuantMSE

**Goal:** Minimize MSE between original and reconstructed vectors.

1. **Rotate**: Apply a random rotation matrix `Pi` (seeded, deterministic).
   - Production path: Randomized Fast Walsh-Hadamard Transform (O(d log d)).
   - Fallback: QR decomposition of a Gaussian matrix (O(d^2)).
2. **Quantize**: For each coordinate of the rotated vector, find the nearest
   centroid in a precomputed Lloyd-Max codebook.
3. **Pack**: Store centroid indices as a packed bit-stream (b bits per coordinate).

Dequantization is the reverse: unpack -> lookup centroids -> inverse rotation.

### Codebook Generation

The codebook is generated via Lloyd-Max quantization over the Beta distribution:

```
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

For high dimensions (d >= 256), this is well-approximated by N(0, 1/d).

Key property: codebooks depend only on (dimension, bit_width), not on the data.

## Algorithm 2: QuantProd

**Goal:** Minimize inner-product estimation error (unbiased estimator).

1. Apply QuantMSE with bit-width `b-1`.
2. Compute residual `r = x - Dequantize(Quantize(x))`.
3. Apply QJL (Quantized Johnson-Lindenstrauss) transform to residual:
   - Project through a random Gaussian matrix.
   - Store only the signs (1-bit sketch) plus residual norm.

The inner product `<x, y>` can be estimated directly from the quantized
representations without full dequantization.

## Key Theorems

**Theorem 1 (MSE bound):** The expected squared error of QuantMSE with b-bit
codebook satisfies a tight bound dependent on the Lloyd-Max distortion for the
Beta(d) distribution.

**Theorem 2 (IP variance):** The QuantProd estimator is unbiased, with variance
bounded by a term that decreases with bit-width.
