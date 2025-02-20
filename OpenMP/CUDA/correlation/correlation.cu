#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define POLYBENCH_TIME 1



/* Include polybench common header. */
#include <polybench.h>
#include <polybench.c>
#define POLYBENCH_TIME 1

typedef double DATA_TYPE; // Definizione di DATA_TYPE

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1000. */
//#define MINI_DATASET
//#define SMALL_DATASET
//#define STANDARD_DATASET
//#define LARGE_DATASET
//#define EXTRALARGE_DATASET

#include "correlation.cuh"

/* Dimensioni predefinite */
#define BLOCK_SIZE 32  // Dimensione blocchi

__global__ void compute_mean(int m, int n, DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE float_n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += data[i * m + j];
        }
        mean[j] = sum / float_n;
    }
}

__global__ void compute_stddev(int m, int n, DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE float_n, DATA_TYPE eps) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < n; i++) {
            DATA_TYPE diff = data[i * m + j] - mean[j];
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / float_n);
        stddev[j] = (stddev[j] <= eps) ? 1.0 : stddev[j];
    }
}

__global__ void normalize_data(int m, int n, DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE float_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        data[i * m + j] = (data[i * m + j] - mean[j]) / (sqrt(float_n) * stddev[j]);
    }
}

__global__ void compute_correlation(int m, int n, DATA_TYPE* data, DATA_TYPE* symmat) {
    int j1 = blockIdx.x * blockDim.x + threadIdx.x;
    int j2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (j1 < m && j2 < m && j1 <= j2) {
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += data[i * m + j1] * data[i * m + j2];
        }
        symmat[j1 * m + j2] = sum;
        symmat[j2 * m + j1] = sum;  // Simmetrico
    }
}

/* Inizializzazione dell'array */
static void init_array(int m, int n, DATA_TYPE *float_n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n)) {
    *float_n = 1.2;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            data[i][j] = ((DATA_TYPE)i * j) / M;
}

/* Stampa della matrice */
static void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m)) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++)
            printf("%f ", symmat[i][j]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    /* Recupera le dimensioni del problema. */
    int n = N;
    int m = M;

    /* Dichiarazione/allocazione variabili. */
    DATA_TYPE float_n;
    POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
    POLYBENCH_2D_ARRAY_DECL(symmat, DATA_TYPE, M, M, m, m);
    POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
    POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);

    /* Inizializzazione degli array. */
    init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

    /* Allocazione GPU */
    DATA_TYPE *d_data, *d_symmat, *d_mean, *d_stddev;
    cudaMalloc((void**)&d_data, m * n * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_symmat, m * m * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_mean, m * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_stddev, m * sizeof(DATA_TYPE));

    /* Copia dati su GPU */
    cudaMemcpy(d_data, POLYBENCH_ARRAY(data), m * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    /* Configurazione dei kernel */
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim1D((m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridDim2D((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Start timer. */
    polybench_start_instruments;

    /* Lancio kernel per media */
    compute_mean<<<gridDim1D, blockDim>>>(m, n, d_data, d_mean, float_n);
    cudaDeviceSynchronize();

    /* Lancio kernel per deviazione standard */
    compute_stddev<<<gridDim1D, blockDim>>>(m, n, d_data, d_mean, d_stddev, float_n, 0.1);
    cudaDeviceSynchronize();

    /* Lancio kernel per normalizzazione */
    normalize_data<<<gridDim2D, blockDim>>>(m, n, d_data, d_mean, d_stddev, float_n);
    cudaDeviceSynchronize();

    /* Lancio kernel per calcolo correlazione */
    compute_correlation<<<gridDim2D, blockDim>>>(m, n, d_data, d_symmat);
    cudaDeviceSynchronize();

    /* Copia risultati su CPU */
    cudaMemcpy(POLYBENCH_ARRAY(symmat), d_symmat, m * m * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
    by the function call in argument. */
    polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));
    //print_array(m, POLYBENCH_ARRAY(symmat));

    /* Pulizia memoria */
    POLYBENCH_FREE_ARRAY(data);
    POLYBENCH_FREE_ARRAY(symmat);
    POLYBENCH_FREE_ARRAY(mean);
    POLYBENCH_FREE_ARRAY(stddev);
    cudaFree(d_data);
    cudaFree(d_symmat);
    cudaFree(d_mean);
    cudaFree(d_stddev);

    return 0;
}