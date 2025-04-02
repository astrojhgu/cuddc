typedef short int int16_t;

struct fcomplex
{
    float real;
    float imag;
};

// DDC 处理所需的 GPU 资源
struct DDCResources
{
    int N; // 每次追加的数据长度
    int M; // 累积多少块数据后计算
    int NDEC;
    int K;
    int16_t *d_indata;
    struct fcomplex *d_outdata;
    struct fcomplex *gpu_buffer;
    float *d_fir_coeffs;
    int16_t *h_indata;
    int h_index;
};

#ifdef __cplusplus
extern "C"
{
#endif
    void init_ddc_resources(struct DDCResources *res, int N, int M, int NDEC, int K, const float *fir_coeffs);
    void free_ddc_resources(struct DDCResources *res);
    int ddc(const int16_t *indata, int lo_ch, struct DDCResources *res);
    void fetch_output(struct fcomplex *outdata, struct DDCResources *res);
    int calc_output_size(struct DDCResources *res);
#ifdef __cplusplus
}
#endif
