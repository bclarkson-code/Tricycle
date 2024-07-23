#include <stdio.h>
#include <cublas_v2.h>
#include <cublasLt.h>

int main() {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        return 1;
    }
    printf("cuBLAS initialized successfully\n");
    cublasDestroy(handle);
    return 0;
}
