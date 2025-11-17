#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstring>

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
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
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << path << std::endl;
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
        std::cerr << "Cannot open file: " << path << std::endl;
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
        std::cerr << "Cannot open parameter file: " << path << std::endl;
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
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================

// CUDA Kernel implementations
__global__ void conv2d_kernel_batched(const float* input,
                                      const float* weight,
                                      const float* bias,
                                      float*       output,
                                      int          N,
                                      int          in_channels,
                                      int          out_channels,
                                      int          input_height,
                                      int          input_width,
                                      int          kernel_size,
                                      int          stride,
                                      int          padding,
                                      int          output_height,
                                      int          output_width) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x; // 输出列
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // 输出行

    if (ow >= output_width || oh >= output_height) {
        return;
    }

    // 2. 用 grid.z 编码 batch 维和 out_channel 维
    //    nz = n * out_channels + oc
    int nz = blockIdx.z;
    int n  = nz / out_channels; // 第几个样本
    int oc = nz % out_channels; // 第几个输出通道

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = ((n * in_channels + ic) * input_height + ih) * input_width + iw;
                    int weight_idx =
                      ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias) {
        sum += bias[oc];
    }

    int output_idx     = ((n * out_channels + oc) * output_height + oh) * output_width + ow;
    output[output_idx] = sum;
}

__global__ void maxpool2d_kernel_batched(const float* input,
                                         float*       output,
                                         int          N,
                                         int          channels,
                                         int          input_height,
                                         int          input_width,
                                         int          kernel_size,
                                         int          stride,
                                         int          output_height,
                                         int          output_width) {
    int out_size = output_height * output_width;
    int total    = N * channels * out_size;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int n   = idx / (channels * out_size);
    int rem = idx % (channels * out_size);
    int c   = rem / out_size;
    int pos = rem % out_size;
    int oh  = pos / output_width;
    int ow  = pos % output_width;

    float max_val = -1e9f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            if (ih < input_height && iw < input_width) {
                int input_idx = ((n * channels + c) * input_height + ih) * input_width + iw;
                max_val       = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    int output_idx     = ((n * channels + c) * output_height + oh) * output_width + ow;
    output[output_idx] = max_val;
}

__global__ void linear_kernel_batched(const float* input,
                                      const float* weight,
                                      const float* bias,
                                      float*       output,
                                      int          N,
                                      int          in_features,
                                      int          out_features) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total)
        return;

    int n = idx / out_features; // 批次索引
    int o = idx % out_features; // 输出特征索引

    float sum         = 0.0f;
    int   input_base  = n * in_features;
    int   weight_base = o * in_features;

    for (int i = 0; i < in_features; ++i) {
        sum += input[input_base + i] * weight[weight_base + i];
    }

    if (bias) {
        sum += bias[o];
    }

    output[idx] = sum;
}

__global__ void
ifnode_kernel(float* input, float* membrane, float* output, int size, float threshold, float reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float v       = membrane[idx] + input[idx];
    float s       = (v >= threshold) ? 1.0f : 0.0f;
    membrane[idx] = (1.0f - s) * v;
    output[idx]   = s;
}

__global__ void accumulate_kernel_batched(const float* input, float* accumulator, int N, int size) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * size;
    if (idx >= total)
        return;
    accumulator[idx] += input[idx];
}

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

    // SNN参数
    const int   T    = 8;
    const float V_TH = 1.0f;

    // 网络参数 - 与正确代码保持一致
    const int in_c = 1, in_h = 28, in_w = 28;
    const int k1 = 5, c1 = 6, c1_h = in_h - k1 + 1, c1_w = in_w - k1 + 1;  // 24x24
    const int p1_h = c1_h / 2, p1_w = c1_w / 2;                            // 12x12
    const int k2 = 5, c2 = 16, c2_h = p1_h - k2 + 1, c2_w = p1_w - k2 + 1; // 8x8
    const int p2_h = c2_h / 2, p2_w = c2_w / 2;                            // 4x4
    const int flat_dim = c2 * p2_h * p2_w;                                 // 256
    const int fc1_out  = 120;
    const int fc2_out  = 84;
    const int fc3_out  = 10;

    // 批次大小
    const int N = 512;

    // 分配设备内存 - 批量版本
    float *d_imgs, *d_conv1_out, *d_conv1_spike, *d_pool1_out;
    float *d_conv2_out, *d_conv2_spike, *d_pool2_out;
    float *d_fc1_out, *d_fc1_spike, *d_fc2_out, *d_fc2_spike, *d_fc3_out;
    float* d_logits_sum; // 用于累积FC3输出的缓冲区

    // 膜电位
    float *d_v1, *d_v2, *d_v3, *d_v4;

    // 计算内存大小
    size_t imgs_size = (size_t)N * in_c * in_h * in_w * sizeof(float);
    size_t c1_size   = (size_t)N * c1 * c1_h * c1_w * sizeof(float);
    size_t p1_size   = (size_t)N * c1 * p1_h * p1_w * sizeof(float);
    size_t c2_size   = (size_t)N * c2 * c2_h * c2_w * sizeof(float);
    size_t p2_size   = (size_t)N * c2 * p2_h * p2_w * sizeof(float);
    size_t fc1_size  = (size_t)N * fc1_out * sizeof(float);
    size_t fc2_size  = (size_t)N * fc2_out * sizeof(float);
    size_t fc3_size  = (size_t)N * fc3_out * sizeof(float);

    // 分配内存
    checkCudaErrors(cudaMalloc(&d_imgs, imgs_size));
    checkCudaErrors(cudaMalloc(&d_conv1_out, c1_size));
    checkCudaErrors(cudaMalloc(&d_conv1_spike, c1_size));
    checkCudaErrors(cudaMalloc(&d_pool1_out, p1_size));
    checkCudaErrors(cudaMalloc(&d_conv2_out, c2_size));
    checkCudaErrors(cudaMalloc(&d_conv2_spike, c2_size));
    checkCudaErrors(cudaMalloc(&d_pool2_out, p2_size));
    checkCudaErrors(cudaMalloc(&d_fc1_out, fc1_size));
    checkCudaErrors(cudaMalloc(&d_fc1_spike, fc1_size));
    checkCudaErrors(cudaMalloc(&d_fc2_out, fc2_size));
    checkCudaErrors(cudaMalloc(&d_fc2_spike, fc2_size));
    checkCudaErrors(cudaMalloc(&d_fc3_out, fc3_size));
    checkCudaErrors(cudaMalloc(&d_logits_sum, fc3_size));

    // 分配膜电位
    checkCudaErrors(cudaMalloc(&d_v1, c1_size));
    checkCudaErrors(cudaMalloc(&d_v2, c2_size));
    checkCudaErrors(cudaMalloc(&d_v3, fc1_size));
    checkCudaErrors(cudaMalloc(&d_v4, fc2_size));

    // 线程配置
    const int threads = 256;

    // 按批次处理图像
    for (int base = 0; base < num_images; base += N) {
        int curN = std::min(N, num_images - base);

        // 复制当前批次图像到设备
        std::vector<float> batch_images(curN * in_c * in_h * in_w);
        for (int n = 0; n < curN; ++n) {
            std::memcpy(batch_images.data() + n * in_c * in_h * in_w,
                        images[base + n].data(),
                        in_c * in_h * in_w * sizeof(float));
        }
        checkCudaErrors(cudaMemcpy(d_imgs,
                                   batch_images.data(),
                                   curN * in_c * in_h * in_w * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // 重置膜电位和累积缓冲区
        checkCudaErrors(cudaMemset(d_v1, 0, (size_t)curN * c1 * c1_h * c1_w * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v2, 0, (size_t)curN * c2 * c2_h * c2_w * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v3, 0, (size_t)curN * fc1_out * sizeof(float)));
        checkCudaErrors(cudaMemset(d_v4, 0, (size_t)curN * fc2_out * sizeof(float)));
        checkCudaErrors(cudaMemset(d_logits_sum, 0, (size_t)curN * fc3_out * sizeof(float)));

        // 计算每个层的线程块数量
        int  blocks_conv1 = (curN * c1 * c1_h * c1_w + threads - 1) / threads;
        dim3 block_conv(16, 16);
        dim3 grid_conv1((c1_w + block_conv.x - 1) / block_conv.x,
                        (c1_h + block_conv.y - 1) / block_conv.y,
                        curN * c1);
        int  blocks_pool1 = (curN * c1 * p1_h * p1_w + threads - 1) / threads;
        int  blocks_conv2 = (curN * c2 * c2_h * c2_w + threads - 1) / threads;
        dim3 grid_conv2((c2_w + block_conv.x - 1) / block_conv.x,
                        (c2_h + block_conv.y - 1) / block_conv.y,
                        curN * c2);
        int  blocks_pool2 = (curN * c2 * p2_h * p2_w + threads - 1) / threads;
        int  blocks_fc1   = (curN * fc1_out + threads - 1) / threads;
        int  blocks_fc2   = (curN * fc2_out + threads - 1) / threads;
        int  blocks_fc3   = (curN * fc3_out + threads - 1) / threads;

        // 时间步循环
        for (int t = 0; t < T; ++t) {
            // 1. Conv1层
            conv2d_kernel_batched<<<grid_conv1, block_conv>>>(d_imgs,
                                                              d_conv1_w,
                                                              d_conv1_b,
                                                              d_conv1_out, //输入输出
                                                              curN,
                                                              in_c,
                                                              c1, // N，channels
                                                              in_h,
                                                              in_w,
                                                              k1,
                                                              1,
                                                              0,
                                                              c1_h,
                                                              c1_w //池化参数
            );

            // 2. Conv1后的IF神经元
            ifnode_kernel<<<blocks_conv1, threads>>>(
              d_conv1_out, d_v1, d_conv1_spike, curN * c1 * c1_h * c1_w, V_TH, 0.0f);

            // 3. Pool1层
            maxpool2d_kernel_batched<<<blocks_pool1, threads>>>(
              d_conv1_spike,
              d_pool1_out, // 输入、输出
              curN,
              c1,
              c1_h,
              c1_w, // N, channels, input_height, input_width
              2,
              2,
              p1_h,
              p1_w // 池化参数
            );

            // 4. Conv2层
            conv2d_kernel_batched<<<grid_conv2, block_conv>>>(d_pool1_out,
                                                              d_conv2_w,
                                                              d_conv2_b,
                                                              d_conv2_out,
                                                              curN,
                                                              c1,
                                                              c2,
                                                              p1_h,
                                                              p1_w,
                                                              k2,
                                                              1,
                                                              0,
                                                              c2_h,
                                                              c2_w);

            // 5. Conv2后的IF神经元
            ifnode_kernel<<<blocks_conv2, threads>>>(
              d_conv2_out, d_v2, d_conv2_spike, curN * c2 * c2_h * c2_w, V_TH, 0.0f);

            // 6. Pool2层
            maxpool2d_kernel_batched<<<blocks_pool2, threads>>>(
              d_conv2_spike, d_pool2_out, curN, c2, c2_h, c2_w, 2, 2, p2_h, p2_w);

            // 7. FC1层 - 直接将池化输出作为扁平化输入
            linear_kernel_batched<<<blocks_fc1, threads>>>(
              d_pool2_out, d_fc1_w, d_fc1_b, d_fc1_out, curN, flat_dim, fc1_out);

            // 8. FC1后的IF神经元
            ifnode_kernel<<<blocks_fc1, threads>>>(
              d_fc1_out, d_v3, d_fc1_spike, curN * fc1_out, V_TH, 0.0f);

            // 9. FC2层
            linear_kernel_batched<<<blocks_fc2, threads>>>(
              d_fc1_spike, d_fc2_w, d_fc2_b, d_fc2_out, curN, fc1_out, fc2_out);

            // 10. FC2后的IF神经元
            ifnode_kernel<<<blocks_fc2, threads>>>(
              d_fc2_out, d_v4, d_fc2_spike, curN * fc2_out, V_TH, 0.0f);

            // 11. FC3层（输出层）
            linear_kernel_batched<<<blocks_fc3, threads>>>(
              d_fc2_spike, d_fc3_w, d_fc3_b, d_fc3_out, curN, fc2_out, fc3_out);

            // 12. 累积FC3输出到logits_sum
            accumulate_kernel_batched<<<blocks_fc3, threads>>>(
              d_fc3_out, d_logits_sum, curN, fc3_out);
        }

        // 复制累积结果到主机并计算预测
        std::vector<float> h_logits(curN * fc3_out);
        checkCudaErrors(cudaMemcpy(
          h_logits.data(), d_logits_sum, curN * fc3_out * sizeof(float), cudaMemcpyDeviceToHost));

        // 对每个样本计算预测（取时间步平均后的最大值）
        for (int n = 0; n < curN; ++n) {
            int   pred = 0;
            float best = h_logits[n * fc3_out + 0] / (float)T; // 时间步平均
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

    // 释放所有设备内存
    checkCudaErrors(cudaFree(d_imgs));
    checkCudaErrors(cudaFree(d_conv1_out));
    checkCudaErrors(cudaFree(d_conv1_spike));
    checkCudaErrors(cudaFree(d_pool1_out));
    checkCudaErrors(cudaFree(d_conv2_out));
    checkCudaErrors(cudaFree(d_conv2_spike));
    checkCudaErrors(cudaFree(d_pool2_out));
    checkCudaErrors(cudaFree(d_fc1_out));
    checkCudaErrors(cudaFree(d_fc1_spike));
    checkCudaErrors(cudaFree(d_fc2_out));
    checkCudaErrors(cudaFree(d_fc2_spike));
    checkCudaErrors(cudaFree(d_fc3_out));
    checkCudaErrors(cudaFree(d_logits_sum));
    checkCudaErrors(cudaFree(d_v1));
    checkCudaErrors(cudaFree(d_v2));
    checkCudaErrors(cudaFree(d_v3));
    checkCudaErrors(cudaFree(d_v4));

    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    // Load test data
    auto images = read_mnist_images(dir + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty())
        return 1;

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

    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
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

    // --- 2. Copy constant parameters from host to device ---
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

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // ===================================================================================
    // Main Function -  DO NOT MODIFY END
    // ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
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

    // ===================================================================================
    // Main Function -  DO NOT MODIFY BEGIN
    // ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    auto                          end  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // --- 4. Free all allocated GPU memory ---
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

    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();

    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;

    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================