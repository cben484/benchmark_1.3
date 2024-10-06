#include "../include/cuda_utils_check.hpp"
#include "cublas_v2.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>

int curandSgenerate(float *matrx, int m, int n, unsigned long long seed);
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed);

int main() {

  cusolverDnHandle_t cusolver_handle;
  CHECK_Cusolver(cusolverDnCreate(&cusolver_handle));

  float *SA;
  float *SWorkspace;
  double *DA;
  double *DWorkspace;
  int SIZE = 7;
  int SLwork;
  int *SdevIpiv;
  int *SdevInfo;
  int DLwork;
  int *DdevIpiv;
  int *DdevInfo;

  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess) {
    printf("Failed to create start event\n");
    return EXIT_SUCCESS;
  }

  if (cudaEventCreate(&stop) != cudaSuccess) {
    printf("Failed to create stop event\n");
    CHECK_Runtime(cudaEventDestroy(start));
    return EXIT_SUCCESS;
  }

  // Single LU分解
  CHECK_Runtime(cudaMalloc((void **)&SA, sizeof(float) * SIZE * SIZE));
  curandSgenerate(SA, SIZE, SIZE, 1234ULL);
  CHECK_Cusolver(cusolverDnSgetrf_bufferSize(cusolver_handle, SIZE, SIZE, SA,
                                             SIZE, &SLwork));

  CHECK_Runtime(cudaMalloc((void **)&SWorkspace, SLwork));
  if (SWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for SWorkspace\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SdevInfo, sizeof(int)));
  if (SdevInfo == nullptr) {
    fprintf(stderr, "Memory allocation failed for SdevInfo\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SdevIpiv, sizeof(int)));
  if (SdevIpiv == nullptr) {
    fprintf(stderr, "Memory allocation failed for SdevIpiv\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnSgetrf(cusolver_handle, SIZE, SIZE, SA, SIZE,
                                  SWorkspace, SdevIpiv, SdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float SelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&SelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnSgetrf execution time: %fms   %fs\n", SelapsedTime,
         SelapsedTime / 1000);

  // Double LU分解
  CHECK_Runtime(cudaMalloc((void **)&DA, sizeof(double) * SIZE * SIZE));
  curandDgenerate(DA, SIZE, SIZE, 4321ULL);
  CHECK_Cusolver(cusolverDnDgetrf_bufferSize(cusolver_handle, SIZE, SIZE, DA,
                                             SIZE, &DLwork));

  CHECK_Runtime(cudaMalloc((void **)&DWorkspace, DLwork));
  if (DWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for DWorkspace\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&DdevInfo, sizeof(int)));
  if (DdevInfo == nullptr) {
    fprintf(stderr, "Memory allocation failed for DdevInfo\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&DdevIpiv, sizeof(int)));
  if (DdevIpiv == nullptr) {
    fprintf(stderr, "Memory allocation failed for DdevIpiv\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnDgetrf(cusolver_handle, SIZE, SIZE, DA, SIZE,
                                  DWorkspace, DdevIpiv, DdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float DelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&DelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnDgetrf execution time: %fms   %fs\n", DelapsedTime,
         DelapsedTime / 1000);
}

// 生成Double
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, matrx, Sum);

  return EXIT_SUCCESS;
}
// 生成Single
int curandSgenerate(float *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, matrx, Sum);

  return EXIT_SUCCESS;
}