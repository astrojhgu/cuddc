#include <iostream>
#include <vector>
#include <complex>
#include <cassert>
#include <fstream>
#include <random>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio> // 使用 C++ 风格的头文件

using namespace std;

constexpr int N = 4096;  // 每次追加的数据长度
constexpr int M = 16384; // 累积多少块数据后计算
constexpr float PI = 3.14159265358979323846;
// DDC 处理所需的 GPU 资源
struct DDCResources
{
    float *d_indata;
    cuFloatComplex *d_outdata;
    cuFloatComplex *gpu_buffer;
    float *d_lo_cos;
    float *d_lo_sin;
    float *d_fir_coeffs;
    int NDEC;
    int K;
    float *h_indata;
    int h_index;
};

// 复数乘法
__device__ cuFloatComplex complex_mult(float a, float b, float c, float d)
{
    return make_cuFloatComplex(a * c - b * d, a * d + b * c);
}

// 设备核函数：执行混频
__global__ void mix(float *indata, float *lo_cos, float *lo_sin, cuFloatComplex *gpu_buffer, int offset, int total_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size)
    {
        gpu_buffer[offset + i] = complex_mult(indata[i], 0.0f, lo_cos[i % N], lo_sin[i % N]);
    }
}

// 设备核函数：FIR 滤波并下抽样
__global__ void fir_filter(cuFloatComplex *gpu_buffer, cuFloatComplex *outdata, float *fir_coeffs, int NDEC, int K, int total_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_index = i;
    int input_index = i * NDEC;

    if (output_index < total_size / NDEC)
    {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int j = 0; j < K * NDEC; j++)
        {
            sum = cuCaddf(sum, cuCmulf(make_cuFloatComplex(fir_coeffs[j], 0.0f), gpu_buffer[input_index + j]));
        }
        outdata[output_index] = sum;
    }
}

// 初始化 DDC 资源
extern "C" void init_ddc_resources(DDCResources *res, int NDEC, int K, float *lo_cos, float *lo_sin, float *fir_coeffs)
{
    res->NDEC = NDEC;
    res->K = K;
    int buffer_size = M * N + NDEC * (K - 1);
    int fir_size = NDEC * K;

    cudaError_t err = cudaMalloc((void **)&res->d_indata, M * N * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_outdata, (M * N / NDEC) * sizeof(cuFloatComplex));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->gpu_buffer, buffer_size * sizeof(cuFloatComplex));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_lo_cos, N * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_lo_sin, N * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_fir_coeffs, fir_size * sizeof(float));
    assert(err == cudaSuccess);

    res->h_indata = (float *)malloc(M * N * sizeof(float));
    assert(res->h_indata);
    res->h_index = 0;

    err = cudaMemcpy(res->d_lo_cos, lo_cos, N * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(res->d_lo_sin, lo_sin, N * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(res->d_fir_coeffs, fir_coeffs, fir_size * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
}

// 释放资源
extern "C" void free_ddc_resources(DDCResources *res)
{
    cudaFree(res->d_indata);
    cudaFree(res->d_outdata);
    cudaFree(res->gpu_buffer);
    cudaFree(res->d_lo_cos);
    cudaFree(res->d_lo_sin);
    cudaFree(res->d_fir_coeffs);
    free(res->h_indata);
}

// DDC 处理
extern "C" int ddc(float *indata, std::complex<float> *outdata, DDCResources *res)
{
    memcpy(res->h_indata + res->h_index, indata, N * sizeof(float));
    res->h_index += N;

    if (res->h_index == M * N)
    {
        int total_size = M * N;
        int buffer_size = total_size + res->NDEC * (res->K - 1);
        int offset = res->NDEC * (res->K - 1);

        cudaMemcpy(res->d_indata, res->h_indata, total_size * sizeof(float), cudaMemcpyHostToDevice);
        mix<<<(total_size + 255) / 256, 256>>>(res->d_indata, res->d_lo_cos, res->d_lo_sin, res->gpu_buffer, offset, total_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;

        fir_filter<<<(total_size / res->NDEC + 255) / 256, 256>>>(res->gpu_buffer, res->d_outdata, res->d_fir_coeffs, res->NDEC, res->K, total_size);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;

        cudaMemcpy(outdata, res->d_outdata, (total_size / res->NDEC) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        res->h_index = 0;
        return 1;
    }
    return 0;
}

int main()
{
    DDCResources res;
    float fir_coeffs[]{-8.30829935e-05, -1.00714599e-04, -1.17997389e-04, -1.34282160e-04,
                       -1.48821308e-04, -1.60767935e-04, -1.69176312e-04, -1.73003787e-04,
                       -1.71114210e-04, -1.62282942e-04, -1.45203459e-04, -1.18495597e-04,
                       -8.07153875e-05, -3.03664859e-05, 3.40868919e-05, 1.14205597e-04,
                       2.11559855e-04, 3.27712375e-04, 4.64200435e-04, 6.22517104e-04,
                       8.04091808e-04, 1.01027046e-03, 1.24229535e-03, 1.50128515e-03,
                       1.78821509e-03, 2.10389778e-03, 2.44896483e-03, 2.82384948e-03,
                       3.22877068e-03, 3.66371865e-03, 4.12844236e-03, 4.62243896e-03,
                       5.14494559e-03, 5.69493352e-03, 6.27110495e-03, 6.87189251e-03,
                       7.49546162e-03, 8.13971567e-03, 8.80230422e-03, 9.48063398e-03,
                       1.01718828e-02, 1.08730165e-02, 1.15808079e-02, 1.22918593e-02,
                       1.30026263e-02, 1.37094443e-02, 1.44085572e-02, 1.50961464e-02,
                       1.57683630e-02, 1.64213591e-02, 1.70513210e-02, 1.76545022e-02,
                       1.82272565e-02, 1.87660701e-02, 1.92675941e-02, 1.97286744e-02,
                       2.01463811e-02, 2.05180355e-02, 2.08412357e-02, 2.11138784e-02,
                       2.13341797e-02, 2.15006913e-02, 2.16123148e-02, 2.16683120e-02,
                       2.16683120e-02, 2.16123148e-02, 2.15006913e-02, 2.13341797e-02,
                       2.11138784e-02, 2.08412357e-02, 2.05180355e-02, 2.01463811e-02,
                       1.97286744e-02, 1.92675941e-02, 1.87660701e-02, 1.82272565e-02,
                       1.76545022e-02, 1.70513210e-02, 1.64213591e-02, 1.57683630e-02,
                       1.50961464e-02, 1.44085572e-02, 1.37094443e-02, 1.30026263e-02,
                       1.22918593e-02, 1.15808079e-02, 1.08730165e-02, 1.01718828e-02,
                       9.48063398e-03, 8.80230422e-03, 8.13971567e-03, 7.49546162e-03,
                       6.87189251e-03, 6.27110495e-03, 5.69493352e-03, 5.14494559e-03,
                       4.62243896e-03, 4.12844236e-03, 3.66371865e-03, 3.22877068e-03,
                       2.82384948e-03, 2.44896483e-03, 2.10389778e-03, 1.78821509e-03,
                       1.50128515e-03, 1.24229535e-03, 1.01027046e-03, 8.04091808e-04,
                       6.22517104e-04, 4.64200435e-04, 3.27712375e-04, 2.11559855e-04,
                       1.14205597e-04, 3.40868919e-05, -3.03664859e-05, -8.07153875e-05,
                       -1.18495597e-04, -1.45203459e-04, -1.62282942e-04, -1.71114210e-04,
                       -1.73003787e-04, -1.69176312e-04, -1.60767935e-04, -1.48821308e-04,
                       -1.34282160e-04, -1.17997389e-04, -1.00714599e-04, -8.30829935e-05};

    std::vector<float> lo_cos, lo_sin;
    auto freq = 1024.0;

    for (int i = 0; i < N; ++i)
    {
        float phase = -(float)i / (float)N * freq * 2 * PI;
        lo_cos.push_back(std::cos(phase));
        lo_sin.push_back(std::sin(phase));
    }
    init_ddc_resources(&res, 8, 16, lo_cos.data(), lo_sin.data(), fir_coeffs);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0, 1.0);

    std::vector<float> indata(N);

    std::vector<std::complex<float>> outdata(N * M / res.NDEC);
    int cnt = 0;

    for (int i = 0;; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float num = dist(gen);
            indata[j] = num;
        }
        auto err = ddc(indata.data(), outdata.data(), &res);
        assert(err >= 0);
        if (err == 1)
        {
            if (cnt++ == 10)
            {
                break;
            }
        }
    }

    ofstream ofs("filtered.dat");

    for (int i = 0; i < N / res.NDEC; ++i)
    {
        std::cout << outdata[i] << std::endl;
    }

    ofs.write((char *)outdata.data(), outdata.size() * sizeof(float) * 2);

    free_ddc_resources(&res);
}
