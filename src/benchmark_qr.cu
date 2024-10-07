#include "../include/check_device.cuh"
#include "../include/cuda_utils_check.hpp"
#include "cublas_v2.h"
#include <algorithm>
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

int main(int argc, char *argv[]) {

  std::cout << "*************************************************" << std::endl;
  std::cout << "探测设备......" << std::endl;
  CHECK_Device(&argv[0]);

  std::cout << "参数总数：" << std::endl;
  std::cout << argc << std::endl;
  std::cout << "参数检查：" << std::endl;
  std::cout << argv[0] << std::endl;
  std::cout << argv[1] << std::endl;
  std::cout << argv[2] << std::endl;
  std::cout << "参数检查完毕" << std::endl;

  std::cout << "*************************************************" << std::endl;

  std::cout << "此矩阵的参数为：" << argv[1] << " x " << argv[2] << std::endl;
  std::cout << "*************************************************" << std::endl;
  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[2]
            << " 规模的 Single qr 分解" << std::endl;

  // 生成handle、各常用参数以及event
  cusolverDnHandle_t cusolver_handle;
  CHECK_Cusolver(cusolverDnCreate(&cusolver_handle));

  float *SA;
  float *SWorkspace;
  double *DA;
  double *DWorkspace;
  int SIZE = 7;
  int SLwork;
  int *SdevInfo;
  float *STAU;
  double *DTAU;
  int DLwork;
  int *DdevInfo;
  int INPUTM = std::atoi(argv[1]);
  int INPUTN = std::atoi(argv[2]);
  float two_three = 2.0 / 3.0;

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

  // Single qr
  CHECK_Runtime(cudaMalloc((void **)&SA, sizeof(float) * INPUTM * INPUTN));
  curandSgenerate(SA, INPUTM, INPUTN, 1234ULL);
  // Sbuffer_qr
  CHECK_Cusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle, INPUTM, INPUTN,
                                             SA, INPUTM, &SLwork));

  CHECK_Runtime(cudaMalloc((void **)&SWorkspace, SLwork * sizeof(float)));

  if (SWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for SWorkspace\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SdevInfo, sizeof(int)));
  if (SdevInfo == nullptr) {
    fprintf(stderr, "Memory allocation failed for SdevInfo\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(
      cudaMalloc((void **)&STAU, sizeof(float) * std::min(INPUTM, INPUTN)));
  if (STAU == nullptr) {
    fprintf(stderr, "Memory allocation failed for TAU\n");
    exit(EXIT_FAILURE);
  }

  // Sqr
  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnSgeqrf(cusolver_handle, INPUTM, INPUTN, SA, INPUTM,
                                  STAU, SWorkspace, SLwork, SdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float SelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&SelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnSgeqrf execution time: %fms   %fs\n", SelapsedTime,
         SelapsedTime / 1000);
  // 输出Single TFLOPS
  std::cout << "the TFLOPS of DnSgetrf is : "
            << (two_three * INPUTM * INPUTN * INPUTN) /
                   ((SelapsedTime / 1000) * 1e12)
            << std::endl;
  std::cout << "*************************************************" << std::endl;

  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[2]
            << " 规模的 Double qr 分解" << std::endl;
  // Double qr
  CHECK_Runtime(cudaMalloc((void **)&DA, sizeof(double) * INPUTM * INPUTN));
  curandDgenerate(DA, INPUTM, INPUTN, 1234ULL);
  // Dbuffer_qr
  CHECK_Cusolver(cusolverDnDgeqrf_bufferSize(cusolver_handle, INPUTM, INPUTN,
                                             DA, INPUTM, &DLwork));

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
  CHECK_Runtime(
      cudaMalloc((void **)&DTAU, sizeof(double) * std::min(INPUTM, INPUTN)));
  if (DTAU == nullptr) {
    fprintf(stderr, "Memory allocation failed for D`AU\n");
    exit(EXIT_FAILURE);
  }

  // Dqr
  CHECK_Runtime(cudaEventRecord(start));
  CHECK_Cusolver(cusolverDnDgeqrf(cusolver_handle, INPUTM, INPUTN, DA, INPUTM,
                                  DTAU, DWorkspace, DLwork, DdevInfo));
  CHECK_Runtime(cudaEventRecord(stop));
  CHECK_Runtime(cudaEventSynchronize(stop));
  // 作差求elapse
  float DelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&DelapsedTime, start, stop));
  // 输出elapse
  printf("\n DnDgeqrf execution time: %fms   %fs\n", DelapsedTime,
         DelapsedTime / 1000);
  // 输出Single TFLOPS
  std::cout << "the TFLOPS of DnDgetrf is : "
            << (two_three * INPUTM * INPUTN * INPUTN) /
                   ((SelapsedTime / 1000) * 1e12)
            << std::endl;
  std::cout << "*************************************************" << std::endl;
  return EXIT_SUCCESS;
}

// 生成Double随机数
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, matrx, Sum);

  return EXIT_SUCCESS;
}
// 生成Single随机数
int curandSgenerate(float *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, matrx, Sum);

  return EXIT_SUCCESS;
}