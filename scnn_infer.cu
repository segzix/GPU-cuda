#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

// ===================================================================================
// Helper for CUDA Error Handling - mirroring template
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// ===================================================================================
// Data and Parameter Loading Functions - mirroring template
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4);
    num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4);
    num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4);
    num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char>      buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4);
    num_items = __builtin_bswap32(num_items);
    std::vector<int>           labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for (int i = 0; i < num_items; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    std::vector<float> params;
    float              param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

// ===================================================================================
// CUDA kernels for layers (simple baseline + batched/fused optimized variants)
// ===================================================================================
// IFNode update with reset-to-zero when crossing threshold (baseline, no batch)
__global__ void if_update_kernel(const float* __restrict__ input_current,
                                 float* __restrict__ v_mem,
                                 float* __restrict__ spike_out,
                                 int   n,
                                 float v_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float v = v_mem[idx] + input_current[idx];
    float s = (v >= v_threshold) ? 1.0f : 0.0f;
    // reset-to-zero: keep v if no spike, zero it if spike
    v_mem[idx]     = (1.0f - s) * v;
    spike_out[idx] = s;
}

// Baseline conv (no batch)
__global__ void conv2d_valid_kernel(const float* __restrict__ x,
                                    int in_c,
                                    int in_h,
                                    int in_w,
                                    const float* __restrict__ w,
                                    const float* __restrict__ b,
                                    int out_c,
                                    int k_h,
                                    int k_w,
                                    float* __restrict__ y) {
    int out_h = in_h - k_h + 1;
    int out_w = in_w - k_w + 1;
    int total = out_c * out_h * out_w;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int oc  = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int oh  = rem / out_w;
    int ow  = rem % out_w;

    float sum = 0.0f;
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int   x_h   = oh + kh;
                int   x_w   = ow + kw;
                float xv    = x[(ic * in_h + x_h) * in_w + x_w];
                int   w_idx = (((oc * in_c) + ic) * k_h + kh) * k_w + kw;
                float wv    = w[w_idx];
                sum += xv * wv;
            }
        }
    }
    y[(oc * out_h + oh) * out_w + ow] = sum + b[oc];
}

// Baseline pool (no batch)
__global__ void maxpool2d_2x2_stride2_kernel(
  const float* __restrict__ x, int channels, int in_h, int in_w, float* __restrict__ y) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int total = channels * out_h * out_w;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int c   = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int oh  = rem / out_w;
    int ow  = rem % out_w;

    int   ih                         = oh * 2;
    int   iw                         = ow * 2;
    int   base                       = (c * in_h + ih) * in_w + iw;
    float a                          = x[base];
    float b                          = x[base + 1];
    float c1                         = x[base + in_w];
    float d                          = x[base + in_w + 1];
    float m1                         = a > b ? a : b;
    float m2                         = c1 > d ? c1 : d;
    y[(c * out_h + oh) * out_w + ow] = m1 > m2 ? m1 : m2;
}

// Baseline flatten (will be removed in optimized path)
__global__ void
flatten_chw_kernel(const float* __restrict__ x, int channels, int h, int w, float* __restrict__ y) {
    int n   = channels * h * w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    y[idx] = x[idx];
}

// Baseline linear (no batch)
__global__ void linear_kernel(const float* __restrict__ x,
                              int in_dim,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              int out_dim,
                              float* __restrict__ y) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim)
        return;
    float sum    = 0.0f;
    int   w_base = o * in_dim;
    for (int i = 0; i < in_dim; ++i) {
        sum += w[w_base + i] * x[i];
    }
    y[o] = sum + b[o];
}

// Device-side accumulation kernel (baseline)
__global__ void
accumulate_kernel(const float* __restrict__ input, float* __restrict__ accumulator, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    accumulator[idx] += input[idx];
}

// ============================
// Batched + Fused kernels
// ============================

// Conv valid + IF fuse, with batch dimension N (layout: NCHW)
__global__ void
conv2d_valid_if_batched_kernel(const float* __restrict__ x,
                               int N,
                               int in_c,
                               int in_h,
                               int in_w,
                               const float* __restrict__ w,
                               const float* __restrict__ b,
                               int out_c,
                               int k_h,
                               int k_w,
                               float* __restrict__ v_mem,  // size N*out_c*out_h*out_w
                               float* __restrict__ spikes, // output spikes same shape as v_mem
                               float v_threshold) {
    int       out_h = in_h - k_h + 1;
    int       out_w = in_w - k_w + 1;
    long long total = (long long)N * out_c * out_h * out_w;
    long long idx   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int       ow = idx % out_w;
    long long t1 = idx / out_w;
    int       oh = t1 % out_h;
    long long t2 = t1 / out_h;
    int       oc = t2 % out_c;
    int       n  = t2 / out_c;

    float sum = 0.0f;
    for (int ic = 0; ic < in_c; ++ic) {
#pragma unroll
        for (int kh = 0; kh < k_h; ++kh) {
            int x_h = oh + kh;
#pragma unroll
            for (int kw = 0; kw < k_w; ++kw) {
                int       x_w   = ow + kw;
                long long x_idx = (((long long)n * in_c + ic) * in_h + x_h) * in_w + x_w;
                int       w_idx = (((oc * in_c) + ic) * k_h + kh) * k_w + kw;
                float     xv    = __ldg(&x[x_idx]);
                float     wv    = __ldg(&w[w_idx]);
                sum += xv * wv;
            }
        }
    }
    sum += __ldg(&b[oc]);

    long long out_idx = (((long long)n * out_c + oc) * out_h + oh) * out_w + ow;
    float     v       = v_mem[out_idx] + sum;
    float     s       = (v >= v_threshold) ? 1.0f : 0.0f;
    v_mem[out_idx]    = (1.0f - s) * v;
    spikes[out_idx]   = s;
}

// MaxPool 2x2 s2 with batch dimension (NCHW)
__global__ void maxpool2d_2x2_stride2_batched_kernel(
  const float* __restrict__ x, int N, int channels, int in_h, int in_w, float* __restrict__ y) {
    int       out_h = in_h / 2;
    int       out_w = in_w / 2;
    long long total = (long long)N * channels * out_h * out_w;
    long long idx   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int       ow = idx % out_w;
    long long t1 = idx / out_w;
    int       oh = t1 % out_h;
    long long t2 = t1 / out_h;
    int       c  = t2 % channels;
    int       n  = t2 / channels;

    int       ih   = oh * 2;
    int       iw   = ow * 2;
    long long base = (((long long)n * channels + c) * in_h + ih) * in_w + iw;
    float     a    = x[base];
    float     b    = x[base + 1];
    float     c1   = x[base + in_w];
    float     d    = x[base + in_w + 1];
    float     m1   = a > b ? a : b;
    float     m2   = c1 > d ? c1 : d;
    y[(((long long)n * channels + c) * out_h + oh) * out_w + ow] = m1 > m2 ? m1 : m2;
}

// Linear + IF fuse with batch
__global__ void linear_if_batched_kernel(const float* __restrict__ x,
                                         int N,
                                         int in_dim,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b,
                                         int out_dim,
                                         float* __restrict__ v_mem,  // N*out_dim
                                         float* __restrict__ spikes, // N*out_dim
                                         float v_threshold) {
    long long total = (long long)N * out_dim;
    long long idx   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int o = idx % out_dim;
    int n = idx / out_dim;

    // y_n_o = sum_i w[o,i]*x[n,i] + b[o]
    float     sum    = 0.0f;
    long long x_base = (long long)n * in_dim;
    int       w_base = o * in_dim;
#pragma unroll 8
    for (int i = 0; i < in_dim; ++i) {
        float xv = __ldg(&x[x_base + i]);
        float wv = __ldg(&w[w_base + i]);
        sum += wv * xv;
    }
    sum += __ldg(&b[o]);

    float v     = v_mem[idx] + sum;
    float s     = (v >= v_threshold) ? 1.0f : 0.0f;
    v_mem[idx]  = (1.0f - s) * v;
    spikes[idx] = s;
}

// Linear that directly accumulates into logits_sum (no IF), with batch
__global__ void
linear_accum_batched_kernel(const float* __restrict__ x,
                            int N,
                            int in_dim,
                            const float* __restrict__ w,
                            const float* __restrict__ b,
                            int out_dim,
                            float* __restrict__ logits_sum // N*out_dim (accumulate over timesteps)
) {
    long long total = (long long)N * out_dim;
    long long idx   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int o = idx % out_dim;
    int n = idx / out_dim;

    float     sum    = 0.0f;
    long long x_base = (long long)n * in_dim;
    int       w_base = o * in_dim;
#pragma unroll 8
    for (int i = 0; i < in_dim; ++i) {
        float xv = __ldg(&x[x_base + i]);
        float wv = __ldg(&w[w_base + i]);
        sum += wv * xv;
    }
    sum += __ldg(&b[o]);

    logits_sum[idx] += sum;
}

// ===================================================================================
// Inference implementation (SNN with IF neurons)
// ===================================================================================
std::vector<int> scnn_inference(const std::vector<std::vector<float>>& images,
                                float*                                 d_conv1_w,
                                float*                                 d_conv1_b,
                                float*                                 d_conv2_w,
                                float*                                 d_conv2_b,
                                float*                                 d_fc1_w,
                                float*                                 d_fc1_b,
                                float*                                 d_fc2_w,
                                float*                                 d_fc2_b,
                                float*                                 d_fc3_w,
                                float*                                 d_fc3_b) {
    std::vector<int> predictions;
    const int        num_images = images.size();
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int   T    = 8;
    const float V_TH = 1.0f; // IF threshold, should match training default

    // Shapes
    const int in_c = 1, in_h = 28, in_w = 28;
    const int k1 = 5, c1 = 6, c1_h = in_h - k1 + 1, c1_w = in_w - k1 + 1;  // 24x24
    const int p1_h = c1_h / 2, p1_w = c1_w / 2;                            // 12x12
    const int k2 = 5, c2 = 16, c2_h = p1_h - k2 + 1, c2_w = p1_w - k2 + 1; // 8x8
    const int p2_h = c2_h / 2, p2_w = c2_w / 2;                            // 4x4
    const int flat_dim = c2 * p2_h * p2_w;                                 // 256
    const int fc1_out  = 120;
    const int fc2_out  = 84;
    const int fc3_out  = 10;

    // Choose a batch size to trade L2 locality vs occupancy. 128 works well on L40S.
    const int N = 512;

    // Allocate batched intermediate device buffers (reused across images)
    float* d_imgs = nullptr;                  // N*1*28*28
    float *d_c1 = nullptr, *d_p1 = nullptr;   // N*c1*24*24 and N*c1*12*12
    float *d_c2 = nullptr, *d_p2 = nullptr;   // N*c2*8*8 and N*c2*4*4
    float* d_flat = nullptr;                  // N*256
    float *d_fc1 = nullptr, *d_fc2 = nullptr; // N*120, N*84

    // Membrane potentials for IF nodes (conv1, conv2, fc1, fc2) with batch
    float *d_v1 = nullptr, *d_v2 = nullptr, *d_v3 = nullptr, *d_v4 = nullptr;

    // Device-side accumulation buffer per-sample
    float* d_logits_sum = nullptr; // N*10

    size_t imgs_size  = (size_t)N * in_c * in_h * in_w * sizeof(float);
    size_t c1_size    = (size_t)N * c1 * c1_h * c1_w * sizeof(float);
    size_t p1_size    = (size_t)N * c1 * p1_h * p1_w * sizeof(float);
    size_t c2_size    = (size_t)N * c2 * c2_h * c2_w * sizeof(float);
    size_t p2_size    = (size_t)N * c2 * p2_h * p2_w * sizeof(float);
    size_t flat_size  = (size_t)N * flat_dim * sizeof(float);
    size_t fc1_size   = (size_t)N * fc1_out * sizeof(float);
    size_t fc2_size   = (size_t)N * fc2_out * sizeof(float);
    size_t logit_size = (size_t)N * fc3_out * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_imgs, imgs_size));
    checkCudaErrors(cudaMalloc(&d_c1, c1_size));
    checkCudaErrors(cudaMalloc(&d_p1, p1_size));
    checkCudaErrors(cudaMalloc(&d_c2, c2_size));
    checkCudaErrors(cudaMalloc(&d_p2, p2_size));
    checkCudaErrors(cudaMalloc(&d_flat, flat_size));
    checkCudaErrors(cudaMalloc(&d_fc1, fc1_size));
    checkCudaErrors(cudaMalloc(&d_fc2, fc2_size));

    checkCudaErrors(cudaMalloc(&d_v1, c1_size));
    checkCudaErrors(cudaMalloc(&d_v2, c2_size));
    checkCudaErrors(cudaMalloc(&d_v3, fc1_size));
    checkCudaErrors(cudaMalloc(&d_v4, fc2_size));

    checkCudaErrors(cudaMalloc(&d_logits_sum, logit_size));

    // Thread config
    const int threads = 256;

    // Process the dataset in chunks of N
    for (int base = 0; base < num_images; base += N) {
        int    curN         = std::min(N, num_images - base);
        size_t curImgsBytes = (size_t)curN * in_c * in_h * in_w * sizeof(float);

        // Pack host images into a contiguous pinned buffer to speed H2D copy
        float* h_pinned = nullptr;
        checkCudaErrors(cudaHostAlloc((void**)&h_pinned, curImgsBytes, cudaHostAllocDefault));
        for (int n = 0; n < curN; ++n) {
            std::memcpy(h_pinned + (size_t)n * in_c * in_h * in_w,
                        images[base + n].data(),
                        (size_t)in_c * in_h * in_w * sizeof(float));
        }
        checkCudaErrors(cudaMemcpy(d_imgs, h_pinned, curImgsBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaFreeHost(h_pinned));

        // Zero membrane and accumulation for this chunk
        checkCudaErrors(cudaMemset(d_v1, 0, (size_t)curN * c1 * c1_h * c1_w * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v2, 0, (size_t)curN * c2 * c2_h * c2_w * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v3, 0, (size_t)curN * fc1_out * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v4, 0, (size_t)curN * fc2_out * sizeof(float)));
        checkCudaErrors(cudaMemset(d_logits_sum, 0, (size_t)curN * fc3_out * sizeof(float)));

        int c1_elems   = curN * c1 * c1_h * c1_w;
        int p1_elems   = curN * c1 * p1_h * p1_w;
        int c2_elems   = curN * c2 * c2_h * c2_w;
        int p2_elems   = curN * c2 * p2_h * p2_w;
        int flat_elems = curN * flat_dim;
        int fc1_elems  = curN * fc1_out;
        int fc2_elems  = curN * fc2_out;
        int fc3_elems  = curN * fc3_out;

        int blocks_c1_if  = (c1_elems + threads - 1) / threads;
        int blocks_p1     = (p1_elems + threads - 1) / threads;
        int blocks_c2_if  = (c2_elems + threads - 1) / threads;
        int blocks_p2     = (p2_elems + threads - 1) / threads;
        int blocks_flat   = (flat_elems + threads - 1) / threads;
        int blocks_fc1_if = (fc1_elems + threads - 1) / threads;
        int blocks_fc2_if = (fc2_elems + threads - 1) / threads;
        int blocks_fc3    = (fc3_elems + threads - 1) / threads;

        for (int t = 0; t < T; ++t) {
            // conv1 + IF (batched)
            conv2d_valid_if_batched_kernel<<<blocks_c1_if, threads>>>(
              d_imgs, curN, in_c, in_h, in_w, d_conv1_w, d_conv1_b, c1, k1, k1, d_v1, d_c1, V_TH);

            // pool1 (batched)
            maxpool2d_2x2_stride2_batched_kernel<<<blocks_p1, threads>>>(
              d_c1, curN, c1, c1_h, c1_w, d_p1);

            // conv2 + IF (batched)
            conv2d_valid_if_batched_kernel<<<blocks_c2_if, threads>>>(
              d_p1, curN, c1, p1_h, p1_w, d_conv2_w, d_conv2_b, c2, k2, k2, d_v2, d_c2, V_TH);

            // pool2 (batched)
            maxpool2d_2x2_stride2_batched_kernel<<<blocks_p2, threads>>>(
              d_c2, curN, c2, c2_h, c2_w, d_p2);

            // flatten is a no-op because we keep contiguous NCHW -> N*(C*H*W)
            // just compute linear on d_p2 with appropriate indexing. For simplicity here, copy to
            // flat. In a further optimization, rewrite linear kernels to read strided from d_p2
            // directly. flatten is a logical no-op; directly consume d_p2 as a flat buffer
            // per-sample

            // fc1 + IF (batched)
            linear_if_batched_kernel<<<blocks_fc1_if, threads>>>(
              d_p2, curN, flat_dim, d_fc1_w, d_fc1_b, fc1_out, d_v3, d_fc1, V_TH);

            // fc2 + IF (batched)
            linear_if_batched_kernel<<<blocks_fc2_if, threads>>>(
              d_fc1, curN, fc1_out, d_fc2_w, d_fc2_b, fc2_out, d_v4, d_fc2, V_TH);

            // fc3 + accumulate (batched)
            linear_accum_batched_kernel<<<blocks_fc3, threads>>>(
              d_fc2, curN, fc2_out, d_fc3_w, d_fc3_b, fc3_out, d_logits_sum);
        }

        // Copy logits for this chunk and compute argmax per sample
        std::vector<float> h_logits(curN * fc3_out);
        checkCudaErrors(cudaMemcpy(h_logits.data(),
                                   d_logits_sum,
                                   (size_t)curN * fc3_out * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int n = 0; n < curN; ++n) {
            int   pred = 0;
            float best = h_logits[n * fc3_out + 0] / (float)T;
            for (int j = 1; j < fc3_out; ++j) {
                float v = h_logits[n * fc3_out + j] / (float)T;
                if (v > best) {
                    best = v;
                    pred = j;
                }
            }
            predictions.push_back(pred);
        }
    }

    // free intermediates
    checkCudaErrors(cudaFree(d_imgs));
    checkCudaErrors(cudaFree(d_c1));
    checkCudaErrors(cudaFree(d_p1));
    checkCudaErrors(cudaFree(d_c2));
    checkCudaErrors(cudaFree(d_p2));
    checkCudaErrors(cudaFree(d_flat));
    checkCudaErrors(cudaFree(d_fc1));
    checkCudaErrors(cudaFree(d_fc2));

    checkCudaErrors(cudaFree(d_v1));
    checkCudaErrors(cudaFree(d_v2));
    checkCudaErrors(cudaFree(d_v3));
    checkCudaErrors(cudaFree(d_v4));

    checkCudaErrors(cudaFree(d_logits_sum));

    return predictions;
}

static bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_dir>" << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    // Candidate data roots (to be robust locally)
    std::vector<std::string> roots = {"./data/FashionMNIST/raw",
                                      dir + "/../data/FashionMNIST/raw",
                                      dir + "/../../.." + "/data/FashionMNIST/raw"};
    std::string              img_path, lbl_path;
    for (const auto& r : roots) {
        std::string ip = r + "/t10k-images-idx3-ubyte";
        std::string lp = r + "/t10k-labels-idx1-ubyte";
        if (file_exists(ip) && file_exists(lp)) {
            img_path = ip;
            lbl_path = lp;
            break;
        }
    }
    if (img_path.empty()) {
        std::cerr << "Could not locate FashionMNIST raw files. Checked a few common paths."
                  << std::endl;
        return 1;
    }

    auto images = read_mnist_images(img_path);
    auto labels = read_mnist_labels(lbl_path);
    if (images.empty() || labels.empty())
        return 1;

    // Sanity check: image/label count must match to avoid out-of-bounds and wrong accuracy
    if (images.size() != labels.size()) {
        std::cerr << "Image/label size mismatch: images=" << images.size()
                  << " labels=" << labels.size() << std::endl;
        return 1;
    }

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/weights/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/weights/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/weights/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/weights/conv2.bias.txt");
    auto fc1_w   = read_param(dir + "/weights/fc1.weight.txt");
    auto fc1_b   = read_param(dir + "/weights/fc1.bias.txt");
    auto fc2_w   = read_param(dir + "/weights/fc2.weight.txt");
    auto fc2_b   = read_param(dir + "/weights/fc2.bias.txt");
    auto fc3_w   = read_param(dir + "/weights/fc3.weight.txt");
    auto fc3_b   = read_param(dir + "/weights/fc3.bias.txt");

    if (conv1_w.empty() || conv1_b.empty() || conv2_w.empty() || conv2_b.empty() || fc1_w.empty()
        || fc1_b.empty() || fc2_w.empty() || fc2_b.empty() || fc3_w.empty() || fc3_b.empty()) {
        std::cerr << "Missing or unreadable weight files in: " << dir << std::endl;
        return 1;
    }

    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters on device
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w, fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b, fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w, fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b, fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w, fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b, fc3_b.size() * sizeof(float)));

    // Copy parameters
    checkCudaErrors(cudaMemcpy(
      d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
      d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
      d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
      d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
      cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();

    // Perform inference
    std::vector<int> predictions = scnn_inference(images,
                                                  d_conv1_w,
                                                  d_conv1_b,
                                                  d_conv2_w,
                                                  d_conv2_b,
                                                  d_fc1_w,
                                                  d_fc1_b,
                                                  d_fc2_w,
                                                  d_fc2_b,
                                                  d_fc3_w,
                                                  d_fc3_b);

    checkCudaErrors(cudaDeviceSynchronize());

    auto                          end  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Free parameters
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    // Accuracy
    int correct = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i])
            correct++;
    }
    double accuracy = static_cast<double>(correct) / labels.size();

    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    return 0;
}
