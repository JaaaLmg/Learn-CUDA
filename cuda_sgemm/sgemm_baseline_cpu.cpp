/**
 * cpu 实现
 */

void MY_MMult_baseline(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc)
{

    for (int r = 0; r < m; r++)
    {
        for (int c = 0; c < n; c++)
        {
            float temp = 0.0f;
            for (int i = 0; i < k; i++)
            {
                temp += A[r * lda + i] * B[i * ldb + c];
            }
            C[r * ldc + c] = temp;
        }
    }
}