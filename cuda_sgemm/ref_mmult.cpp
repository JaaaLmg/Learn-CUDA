/* Create macros so that the matrices are stored in row-major order */
#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

/*
 * 本机只安装了 libblas.so.3（导出了 cblas_sgemm 符号），
 * 但缺少开发头文件 cblas.h，因此这里自行声明所需的 CBLAS 枚举与原型。
 * 枚举取值遵循 CBLAS 标准（Reference BLAS / OpenBLAS 一致）。
 */
extern "C"
{
    enum CBLAS_ORDER
    {
        CblasRowMajor = 101,
        CblasColMajor = 102
    };
    enum CBLAS_TRANSPOSE
    {
        CblasNoTrans = 111,
        CblasTrans = 112,
        CblasConjTrans = 113
    };

    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha,
                     const float *A, int lda, const float *B, int ldb,
                     float beta, float *C, int ldc);
}

/* Routine for computing C = A * B + C */

void REF_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb,
               float *c, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda,
                b, ldb, 0.0f, c, ldc);
}