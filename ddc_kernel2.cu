#include <complex>
#include <cassert>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio> // 使用 C++ 风格的头文件

using namespace std;
static constexpr float PI=3.14159265358979323846;


struct DDCResources2 //secondary ddc
{
    int N;  // 每次追加的数据长度
    int NDEC;
    int K;
    cuFloatComplex* d_indata;
    cuFloatComplex *d_outdata;
    cuFloatComplex *gpu_buffer;
    float *d_fir_coeffs;
};



// 复数乘法
static __device__ cuFloatComplex complex_mult(float a, float b, float c, float d)
{
    return make_cuFloatComplex(a * c - b * d, a * d + b * c);
}

__global__ void mix2(cuFloatComplex *indata, cuFloatComplex *gpu_buffer, int offset, int N, int lo_ch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float phase=-(float)i*(float)lo_ch/(float)N*2.0*PI;
        float lo_cos=cos(phase);
        float lo_sin=sin(phase);
        gpu_buffer[offset + i] = complex_mult(indata[i].x, indata[i].y, lo_cos, lo_sin);
    }
}

__global__ void fir_filter2(cuFloatComplex *gpu_buffer, cuFloatComplex *outdata, const float *fir_coeffs, int NDEC, int K, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_index = i;
    int input_index = i * NDEC;

    if (output_index < N / NDEC)
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
extern "C" void init_ddc_resources2(DDCResources2 *res,int N, int NDEC, int K, const float *fir_coeffs)
{
    res->NDEC = NDEC;
    res->K = K;
    res->N=N;
    int buffer_size =  N + NDEC * (K - 1);
    int fir_size = NDEC * K;

    cudaError_t err = cudaMalloc((void **)&res->d_indata, N * sizeof(cuFloatComplex));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_outdata, (N / NDEC) * sizeof(cuFloatComplex));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->gpu_buffer, buffer_size * sizeof(cuFloatComplex));
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&res->d_fir_coeffs, fir_size * sizeof(float));
    assert(err == cudaSuccess);

    err = cudaMemcpy(res->d_fir_coeffs, fir_coeffs, fir_size * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
}

// 释放资源
extern "C" void free_ddc_resources2(DDCResources2 *res)
{
    cudaFree(res->d_indata);
    cudaFree(res->d_outdata);
    cudaFree(res->gpu_buffer);
    cudaFree(res->d_fir_coeffs);
}

// DDC 处理
extern "C" int ddc2(const cuFloatComplex *indata, int lo_ch, DDCResources2 *res)
{

        //int buffer_size = total_size + res->NDEC * (res->K - 1);
        int offset = res->NDEC * (res->K - 1);

        cudaMemcpy(res->d_indata, indata, res->N * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
        mix2<<<(res->N + 255) / 256, 256>>>(res->d_indata, res->gpu_buffer, offset, res->N, lo_ch);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;

        fir_filter2<<<(res->N / res->NDEC + 255) / 256, 256>>>(res->gpu_buffer, res->d_outdata, res->d_fir_coeffs, res->NDEC, res->K, res->N);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return -1;
        return 1;
}

extern "C" void fetch_output2(std::complex<float> *outdata, DDCResources2* res){
    cudaMemcpy(outdata, res->d_outdata, (res->N / res->NDEC) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
}


extern "C" int calc_output_size2(const DDCResources2* res){
    return (res->N)/(res->NDEC);
}
