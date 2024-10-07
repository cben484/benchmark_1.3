#include "../include/check_device.cuh"
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
#include <string>

int curandSgenerate(float *matrx, int m, int n, unsigned long long seed);
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed);

int main(int argc, char *argv[]) {

  std::cout << "*************************************************" << std::endl;
  std::cout << "探测设备......" << std::endl;
  CHECK_Device(&argv[0]);

  std::cout << "参数总数：" << std::endl;
  std::cout << argc << std::endl;
  std::cout << "参数检查：" << std::endl;
  std::cout << argv[0] << std::endl;
  std::cout << argv[1] << std::endl;
  std::cout << "参数检查完毕" << std::endl;

  std::cout << "*************************************************" << std::endl;

  std::cout << "此方阵的参数为：" << argv[1] << " x " << argv[1] << std::endl;
  std::cout << "*************************************************" << std::endl;
  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[1]
            << " 规模的 Single LU 分解" << std::endl;

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
  float two_three = 2.0 / 3.0;
  int INPUTN = std::stoi(argv[1]);

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
  CHECK_Runtime(cudaMalloc((void **)&SA, sizeof(float) * INPUTN * INPUTN));
  curandSgenerate(SA, INPUTN, INPUTN, 1234ULL);
  CHECK_Cusolver(cusolverDnSgetrf_bufferSize(cusolver_handle, INPUTN, INPUTN,
                                             SA, INPUTN, &SLwork));

  CHECK_Runtime(cudaMalloc((void **)&SWorkspace, sizeof(float) * SLwork));
  if (SWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for SWorkspace\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SdevInfo, sizeof(int)));
  if (SdevInfo == nullptr) {
    fprintf(stderr, "Memory allocation failed for SdevInfo\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SdevIpiv, sizeof(int) * INPUTN));
  if (SdevIpiv == nullptr) {
    fprintf(stderr, "Memory allocation failed for SdevIpiv\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnSgetrf(cusolver_handle, INPUTN, INPUTN, SA, INPUTN,
                                  SWorkspace, SdevIpiv, SdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float SelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&SelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnSgetrf execution time: %fms   %fs\n", SelapsedTime,
         SelapsedTime / 1000);
  // 输出Single TFLOPS
  std::cout << "the TFLOPS of DnSgetrf is : "
            << (two_three * INPUTN * INPUTN * INPUTN) /
                   ((SelapsedTime / 1000) * 1e12)
            << std::endl;
  std::cout << "*************************************************" << std::endl;

  // Double LU分解
  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[1]
            << " 规模的 Double LU 分解" << std::endl;
  CHECK_Runtime(cudaMalloc((void **)&DA, sizeof(double) * INPUTN * INPUTN));
  curandDgenerate(DA, INPUTN, INPUTN, 4321ULL);
  CHECK_Cusolver(cusolverDnDgetrf_bufferSize(cusolver_handle, INPUTN, INPUTN,
                                             DA, INPUTN, &DLwork));

  CHECK_Runtime(cudaMalloc((void **)&DWorkspace, sizeof(double) * DLwork));
  if (DWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for DWorkspace\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&DdevInfo, sizeof(int)));
  if (DdevInfo == nullptr) {
    fprintf(stderr, "Memory allocation failed for DdevInfo\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&DdevIpiv, sizeof(int) * INPUTN));
  if (DdevIpiv == nullptr) {
    fprintf(stderr, "Memory allocation failed for DdevIpiv\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnDgetrf(cusolver_handle, INPUTN, INPUTN, DA, INPUTN,
                                  DWorkspace, DdevIpiv, DdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float DelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&DelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnDgetrf execution time: %fms   %fs\n", DelapsedTime,
         DelapsedTime / 1000);

  // 输出Single TFLOPS
  std::cout << "the TFLOPS of DnDgetrf is : "
            << (two_three * INPUTN * INPUTN * INPUTN) /
                   ((DelapsedTime / 1000) * 1e12)
            << std::endl;
  std::cout << "*************************************************" << std::endl;
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