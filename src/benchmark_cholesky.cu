#include "../include/check_device.cuh"
#include "../include/cuda_utils_check.hpp"
#include "../include/macro.hpp"
#include "cublas_v2.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iomanip> // std::setprecision
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>

int curandgenerate_gemm(double *matrx, int m, int n, unsigned long long seed);
int validate_ss_D(cublasHandle_t handle, double const *matrix, double *result,
                  double const *origin, int N);
int print_matrix_rowmajor(double *matrix, int m, int n);
template <typename T> int print_matrix_colmajor(T *matrix, int m, int n);
int validate_ss_S(cublasHandle_t handle, float const *matrix, float *result,
                  float const *origin, int N);

int main(int argc, char *argv[]) {

  std::cout << "*************************************************" << std::endl;
  std::cout << "探测设备......" << std::endl;
  CHECK_Device(&argv[0]);

  std::cout << "*************************************************" << std::endl;
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
            << " 规模的 Double Cholesky 分解" << std::endl;

  // 输入的矩阵的规模
  int INPUTN = std::stoi(argv[1]);

  cusolverDnHandle_t cusolver_handle;
  cublasHandle_t cublas_handle;
  CHECK_Cusolver(cusolverDnCreate(&cusolver_handle));
  CHECK_Cublas(cublasCreate(&cublas_handle));
  // Double precision
  double *DA;
  double *DB;
  int DLwork;
  int *DdevInfo;
  double *DWorkspace;
  double Dalpha = 1.0;
  double Dbeta = 0.0;
  double temp = 1.0;
  float one_three = 1.0 / 3.0;

  // 创建显示stream和event（目前此代码中是单stream，和隐式一般无二）
  cudaStream_t stream;
  CHECK_Runtime(cudaStreamCreate(&stream));
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

  // 初始化操作,使用curand为DA随机初始化赋值
  CHECK_Runtime(cudaMalloc((void **)&DA, sizeof(double) * INPUTN * INPUTN));
  if (DA == nullptr) {
    fprintf(stderr, "Memory allocation failed for DA\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&DB, sizeof(double) * INPUTN * INPUTN));
  if (DB == nullptr) {
    fprintf(stderr, "Memory allocation failed for DB\n");
    exit(EXIT_FAILURE);
  }

  curandGenerator_t gen;
  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  CHECK_Curand(curandGenerateUniformDouble(gen, DB, INPUTN * INPUTN));

  // 原本是查看生成的随机矩阵DB：
  double *h_DB = new double[INPUTN * INPUTN];
  CHECK_Runtime(cudaMemcpy(h_DB, DB, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // DB+=I，提高DB的秩，一定程度上确保DB为满秩矩阵
  for (int i = 0; i < INPUTN; ++i) {
    h_DB[i * INPUTN + i] += temp;
  }
  CHECK_Runtime(cudaMemcpy(DB, h_DB, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyHostToDevice));

  // DA=DB^T*DB
  CHECK_Cublas(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUTN,
                           INPUTN, INPUTN, &Dalpha, DB, INPUTN, DB, INPUTN,
                           &Dbeta, DA, INPUTN));

  CHECK_Runtime(cudaMemcpy(h_DB, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // 开始前先给GPU热身
  for (size_t i{0}; i < warmups; ++i) {
    CHECK_Cublas(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUTN,
                             INPUTN, INPUTN, &Dalpha, DB, INPUTN, DB, INPUTN,
                             &Dbeta, DB, INPUTN));
  }

  // 要为A分配内存，但是不必初始化，分配内存主要是因为buffer函数需要一个指针参数，并且buffer函数会以此与lda、n做验证，从而确定需要的Lwork

  // buffer之前检查一下DA的模样，原本是查看DA
  CHECK_Runtime(cudaMemcpy(h_DB, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  CHECK_Cusolver(cusolverDnDpotrf_bufferSize(
      cusolver_handle, CUBLAS_FILL_MODE_LOWER, INPUTN, DA, INPUTN, &DLwork));
  // buffer之后检查一下DA的模样，原本是查看DA
  CHECK_Runtime(cudaMemcpy(h_DB, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  CHECK_Runtime(cudaMalloc((void **)&DWorkspace, DLwork * sizeof(double)));

  if (DWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for DWorkspace\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaMalloc((void **)&DdevInfo, sizeof(int)));
  // Dportf之前检查一下DA的模样，原本是查看DA
  double *h_DB_origin;
  h_DB_origin = (double *)malloc(sizeof(double) * INPUTN * INPUTN);
  CHECK_Runtime(cudaMemcpy(h_DB_origin, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // 记录start
  CHECK_Runtime(cudaEventRecord(start, stream));
  // 要为A和Workspace分配空间，以及初始化A，对A进行potrf

  CHECK_Cusolver(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                  INPUTN, DA, INPUTN, DWorkspace, DLwork,
                                  DdevInfo));
  // 记录stop
  CHECK_Runtime(cudaEventRecord(stop, stream));
  // 同步
  CHECK_Runtime(cudaEventSynchronize(stop));
  CHECK_Runtime(cudaStreamSynchronize(stream));

  // Dportf之后检查一下DA的模样，原本是查看一下DA
  CHECK_Runtime(cudaMemcpy(h_DB, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // 作差求elapse
  float DelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&DelapsedTime, start, stop));
  // 计算latency
  float const Dlatency{DelapsedTime};
  // 计算TFLOPS
  float const Dtflops{(one_three * INPUTN * INPUTN * INPUTN) /
                      ((Dlatency * 1e-3f) * 1e12f)};
  std::cout << "*************************************************" << std::endl;
  // 输出TFLOPS
  std::cout << "双精度Cholesky分解的TFLOPS: " << Dtflops << " TFLOPS"
            << std::endl;
  // std::cout << "Dpotrf Effective TFLOPS:" << Dtflops << "TFLOPS" <<
  // std::endl; 输出elapse
  printf("\n双精度Cholesky分解的执行时间 : %fms   %fs\n", DelapsedTime,
         DelapsedTime / 1000);
  std::cout << "*************************************************" << std::endl;
  // printf("\n Dpotrf execution time: %fms   %fs\n", DelapsedTime,
  //        DelapsedTime / 1000);

  // validate之前检查一下DA的模样，原本是检查一下DA
  CHECK_Runtime(cudaMemcpy(h_DB, DA, sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  double result;
  // 这里传进去的origin要直接放到memcpy里面作为src以device2host的方式，所以这里的origin需要一个device的版本，但是h_DB是host的，所以一直会报错，解决方案：memcpy
  double *h2d_DB;
  CHECK_Runtime(cudaMalloc((void **)&h2d_DB, sizeof(double) * INPUTN * INPUTN));
  CHECK_Runtime(cudaMemcpy(h2d_DB, h_DB_origin,
                           sizeof(double) * INPUTN * INPUTN,
                           cudaMemcpyHostToDevice));
  std::cout << "双精度验证开始" << std::endl;
  validate_ss_D(cublas_handle, DA, &result, h2d_DB, INPUTN);

  std::cout << std::fixed << std::setprecision(20);
  std::cout << "the validate value is:" << result << std::endl;
  std::cout << "双精度验证完毕" << std::endl;
  std::cout << "*************************************************" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // Single precision
  float *SB;
  float *SA;
  int SLwork;
  int *SdevInfo;
  float *SWorkspace;
  float Salpha = 1.0;
  float Sbeta = 0.0;
  float *stemp;

  stemp = (float *)malloc(sizeof(float) * INPUTN * INPUTN);
  CHECK_Runtime(cudaMalloc((void **)&SA, sizeof(float) * INPUTN * INPUTN));
  if (SA == nullptr) {
    fprintf(stderr, "Memory allocation failed for SA\n");
    exit(EXIT_FAILURE);
  }
  CHECK_Runtime(cudaMalloc((void **)&SB, sizeof(float) * INPUTN * INPUTN));
  if (SB == nullptr) {
    fprintf(stderr, "Memory allocation failed for SB\n");
    exit(EXIT_FAILURE);
  }

  // 初始化操作,使用curand为SB随机初始化赋值,利用SB乘以SB的装置得到一个对称但不一定正定矩阵，所以在SB*SB转置之前要对SB做特殊操作，例如SB+αI（α是正数），
  // curandGenerator_t gen; 可以用一个curandGenerator_t随机初始化两个矩阵
  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 4321ULL));
  CHECK_Curand(curandGenerateUniform(gen, SB, INPUTN * INPUTN));

  // SB+=I，提高SB的秩，一定程度上确保SB为满秩矩阵
  float *h_SB = new float[INPUTN * INPUTN];
  CHECK_Runtime(cudaMemcpy(h_SB, SB, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));
  // std::cout << "the SB is:" << std::endl;
  // print_matrix_colmajor(h_SB, INPUTN, INPUTN);
  for (int i{0}; i < INPUTN; ++i) {
    h_SB[i * INPUTN + i] += 1.0f;
  }

  // 看下SB提高秩之后
  // std::cout << "看下SB提高秩之后" << std::endl;
  // print_matrix_colmajor(h_SB, INPUTN, INPUTN);

  CHECK_Runtime(cudaMemcpy(SB, h_SB, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyHostToDevice));

  // A=SB^T*SB
  CHECK_Cublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUTN,
                           INPUTN, INPUTN, &Salpha, SB, INPUTN, SB, INPUTN,
                           &Sbeta, SA, INPUTN));

  CHECK_Runtime(cudaMemcpy(stemp, SA, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));
  // std::cout << "让SB对称正交化:" << std::endl;
  // print_matrix_colmajor(stemp, INPUTN, INPUTN);

  // 开局先热身
  for (size_t i{0}; i < warmups / 2; ++i) {
    CHECK_Cublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUTN,
                             INPUTN, INPUTN, &Salpha, SB, INPUTN, SB, INPUTN,
                             &Sbeta, SB, INPUTN));
  }

  // 要为A分配内存，但是不必初始化，分配内存主要是因为buffer函数需要一个指针参数，并且buffer函数会以此与lda、n做验证，从而确定需要的Lwork
  CHECK_Cusolver(cusolverDnSpotrf_bufferSize(
      cusolver_handle, CUBLAS_FILL_MODE_LOWER, INPUTN, SA, INPUTN, &SLwork));

  CHECK_Runtime(cudaMalloc((void **)&SWorkspace, sizeof(float) * SLwork));
  if (SWorkspace == nullptr) {
    fprintf(stderr, "Memory allocation failed for SWorkspace\n");
    exit(EXIT_FAILURE);
  }

  CHECK_Runtime(cudaMalloc((void **)&(SdevInfo), sizeof(int)));
  float *h_SB_origin;
  h_SB_origin = (float *)malloc(sizeof(float) * INPUTN * INPUTN);
  CHECK_Runtime(cudaMemcpy(h_SB_origin, SA, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // 对start进行record
  CHECK_Runtime(cudaEventRecord(start, stream));
  // 要为A分配内存并且对A进行初始化
  // 为了精确，所以repeat
  // for (size_t i{0}; i < num_repeats; i++) {
  CHECK_Cusolver(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                  INPUTN, SA, INPUTN, SWorkspace, SLwork,
                                  SdevInfo));
  // }
  // 对stop进行record
  CHECK_Runtime(cudaEventRecord(stop, stream));
  // 同步
  CHECK_Runtime(cudaEventSynchronize(stop));
  CHECK_Runtime(cudaStreamSynchronize(stream));

  // 作差求elapse
  float SelapsedTime;
  CHECK_Runtime(cudaEventElapsedTime(&SelapsedTime, start, stop));
  // 计算latency
  float const Slatency{SelapsedTime};
  // 计算TFLOPS
  float const Stflops{(one_three * INPUTN * INPUTN * INPUTN) /
                      ((Slatency * 1e-3f) * 1e12f)};
  std::cout << "*************************************************" << std::endl;

  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[1]
            << " 规模的 Single Cholesky 分解" << std::endl;
  std::cout << "*************************************************" << std::endl;

  // 输出TFLOPS
  std::cout << "单精度Cholesky分解的TFLOPS:" << Stflops << " TFLOPS"
            << std::endl;
  // 输出elapse
  printf("\n单精度Cholesky分解的执行时间: %fms   %fs\n", SelapsedTime,
         SelapsedTime / 1000);
  std::cout << "*************************************************" << std::endl;

  // 查看一下SA和h2d_SB

  CHECK_Runtime(cudaMemcpy(stemp, SA, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyDeviceToHost));

  // std::cout << "对single验证之前查看一下输入矩阵SA" << std::endl;
  // print_matrix_colmajor(stemp, INPUTN, INPUTN);

  CHECK_Runtime(cudaMemcpy(stemp, h_SB_origin, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyHostToHost));
  // std::cout << "对single验证之前查看一下输入矩阵h2d_SB" << std::endl;
  // print_matrix_colmajor(stemp, INPUTN, INPUTN);

  float Sresult;
  // 这里传进去的origin要直接放到memcpy里面作为src以device2host的方式，所以这里的origin需要一个device的版本，但是h_DB是host的，所以一直会报错，解决方案：memcpy
  float *h2d_SB;
  CHECK_Runtime(cudaMalloc((void **)&h2d_SB, sizeof(float) * INPUTN * INPUTN));
  CHECK_Runtime(cudaMemcpy(h2d_SB, h_SB_origin, sizeof(float) * INPUTN * INPUTN,
                           cudaMemcpyHostToDevice));
  validate_ss_S(cublas_handle, SA, &Sresult, h2d_SB, INPUTN);

  std::cout << std::fixed << std::setprecision(20);
  std::cout << "the** Spotrf validate value is:" << Sresult << std::endl;

  // clean

  free(h_DB);
  free(h_SB);

  cusolverDnDestroy(cusolver_handle);
  cublasDestroy(cublas_handle);
  cudaStreamDestroy(stream);
  cudaFree(DA);
  cudaFree(DB);
  cudaFree(DWorkspace);
  cudaFree(SA);
  cudaFree(SB);
  cudaFree(SWorkspace);

  return EXIT_SUCCESS;
}

// validate from ss
int validate_ss_D(cublasHandle_t handle, double const *matrix, double *result,
                  double const *origin, int N) {

  double alpha = -1.0;
  double beta = 1.0;
  // double *numerator; // 分子
  double nresult;
  // double *denominator; // 分母
  double *temp;
  double dresult;
  double *L;
  double *H_L;
  double *H_matrix;
  double *temp_origin;

  CHECK_Runtime(cudaMalloc((void **)&(L), sizeof(double) * N * N));
  CHECK_Runtime(cudaMalloc((void **)&(temp), sizeof(double) * N * N));
  CHECK_Runtime(cudaMalloc((void **)&(temp_origin), sizeof(double) * N * N));

  // 显示检查最后一个是否出问题
  cudaError_t err1 = cudaGetLastError();
  std::cout << "the err1 is:" << err1 << std::endl;

  CHECK_Runtime(cudaMemcpy(temp_origin, origin, sizeof(double) * N * N,
                           cudaMemcpyDeviceToDevice));
  CHECK_Runtime(cudaMemcpy(temp, matrix, sizeof(double) * N * N,
                           cudaMemcpyDeviceToDevice));

  H_matrix = (double *)malloc(sizeof(double) * N * N);
  H_L = (double *)malloc(sizeof(double) * N * N);
  CHECK_Runtime(cudaMemcpy(H_matrix, matrix, sizeof(double) * N * N,
                           cudaMemcpyDeviceToHost));
  // std::cout << "the matrix is :" << std::endl;
  // print_matrix_colmajor(H_matrix, N, N);

  // 下三角
  for (int j = 0; j < N; ++j) { // 遍历列
    // 复制下三角部分的元素，包括对角线
    for (int i = j; i < N; ++i) { // 从对角线开始遍历行
      H_L[i + j * N] = H_matrix[i + j * N];
    }
    // 可选地，将上三角部分（对角线之上）的元素设为零
    for (int i = 0; i < j; ++i) { // 遍历对角线之上的行
      H_L[i + j * N] = 0.0;
    }
  }

  // // 上三角
  // //  H_matrix 和 H_U 是指向 n x n 矩阵的指针，矩阵以列主序存储
  // //  将 H_U 初始化为零（可选）
  // for (int j = 0; j < SIZE; ++j) { // 遍历列
  //   // 复制上三角部分的元素，包括对角线
  //   for (int i = 0; i <= j; ++i) { // 遍历行直到对角线
  //     H_L[i + j * SIZE] = H_matrix[i + j * SIZE];
  //   }
  //   // 可选地，将下三角部分（对角线之下）的元素设为零
  //   for (int i = j + 1; i < SIZE; ++i) { // 遍历对角线之下的行
  //     H_L[i + j * SIZE] = 0.0;
  //   }
  // }

  // std::cout << "the L is :" << std::endl;
  // print_matrix_colmajor(H_L, N, N);
  CHECK_Runtime(
      cudaMemcpy(L, H_L, sizeof(double) * N * N, cudaMemcpyHostToDevice));
  // 上来就算分母A的L2范数
  CHECK_Cublas(cublasDnrm2(handle, N * N, temp_origin, 1, &dresult));
  std::cout << "the dresult is:" << dresult << std::endl;
  // 在syrk之前查看一下temp_origin的模样
  // std::cout << "before syrk the temp_origin is:" << std::endl;
  double *lookthetemporigin;
  lookthetemporigin = (double *)malloc(sizeof(double) * N * N);
  CHECK_Runtime(cudaMemcpy(lookthetemporigin, temp_origin,
                           sizeof(double) * N * N, cudaMemcpyDeviceToHost));

  // print_matrix_colmajor(lookthetemporigin, N, N);
  // 用syrk算出A-LL^T
  CHECK_Cublas(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, N,
                           &alpha, L, N, &beta, temp_origin, N));
  // syrk算出来的结果是对称的，所以syrk只会按照uplo的指示将结果存于下三角或者是上三角，才意识到，所以还需要对其进行一个映射操作，之后再进行nrm2运算

  std::cout << "after syrk the temp_origin is:" << std::endl;
  CHECK_Runtime(cudaMemcpy(lookthetemporigin, temp_origin,
                           sizeof(double) * N * N, cudaMemcpyDeviceToHost));
  // print_matrix_colmajor(lookthetemporigin, N, N);

  // 将下三角映射到上三角，从而构成一个对称矩阵
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      lookthetemporigin[j * N + i] =
          lookthetemporigin[i * N + j]; // 将下三角的值赋给上三角
    }
  }
  // std::cout << "映射情况：" << std::endl;
  // print_matrix_colmajor(lookthetemporigin, N, N);
  double *fenzi;
  CHECK_Runtime(cudaMalloc((void **)&(fenzi), sizeof(double) * N * N));
  CHECK_Runtime(cudaMemcpy(fenzi, lookthetemporigin, sizeof(double) * N * N,
                           cudaMemcpyHostToDevice));
  // 用Dnrm2算分子的L2范数
  CHECK_Cublas(cublasDnrm2(handle, N * N, fenzi, 1, &nresult));
  std::cout << std::fixed << std::setprecision(20);
  std::cout << "the nresult is:" << nresult << std::endl;

  *result = nresult / dresult;

  CHECK_Runtime(cudaFree(L));
  CHECK_Runtime(cudaFree(temp));
  CHECK_Runtime(cudaFree(temp_origin));
  free(H_L);
  free(H_matrix);

  std::cout << "验证完成" << std::endl;

  return EXIT_SUCCESS;
}

int validate_ss_S(cublasHandle_t handle, float const *matrix, float *result,
                  float const *origin, int N) {

  float alpha = -1.0;
  float beta = 1.0;
  // double *numerator; // 分子
  float nresult;
  // double *denominator; // 分母
  float *temp;
  float dresult;
  float *L;
  float *H_L;
  float *H_matrix;
  float *temp_origin;

  CHECK_Runtime(cudaMalloc((void **)&(L), sizeof(float) * N * N));
  CHECK_Runtime(cudaMalloc((void **)&(temp), sizeof(float) * N * N));
  CHECK_Runtime(cudaMalloc((void **)&(temp_origin), sizeof(float) * N * N));

  // 显示检查最后一个是否出问题
  cudaError_t err1 = cudaGetLastError();
  std::cout << "the err1 is:" << err1 << std::endl;

  CHECK_Runtime(cudaMemcpy(temp_origin, origin, sizeof(float) * N * N,
                           cudaMemcpyDeviceToDevice));
  CHECK_Runtime(cudaMemcpy(temp, matrix, sizeof(float) * N * N,
                           cudaMemcpyDeviceToDevice));

  H_matrix = (float *)malloc(sizeof(float) * N * N);
  H_L = (float *)malloc(sizeof(float) * N * N);
  CHECK_Runtime(cudaMemcpy(H_matrix, matrix, sizeof(float) * N * N,
                           cudaMemcpyDeviceToHost));
  // std::cout << "the matrix is :" << std::endl;
  // print_matrix_colmajor(H_matrix, N, N);

  // 下三角
  for (int j = 0; j < N; ++j) { // 遍历列
    // 复制下三角部分的元素，包括对角线
    for (int i = j; i < N; ++i) { // 从对角线开始遍历行
      H_L[i + j * N] = H_matrix[i + j * N];
    }
    // 可选地，将上三角部分（对角线之上）的元素设为零
    for (int i = 0; i < j; ++i) { // 遍历对角线之上的行
      H_L[i + j * N] = 0.0;
    }
  }

  // // 上三角
  // //  H_matrix 和 H_U 是指向 n x n 矩阵的指针，矩阵以列主序存储
  // //  将 H_U 初始化为零（可选）
  // for (int j = 0; j < SIZE; ++j) { // 遍历列
  //   // 复制上三角部分的元素，包括对角线
  //   for (int i = 0; i <= j; ++i) { // 遍历行直到对角线
  //     H_L[i + j * SIZE] = H_matrix[i + j * SIZE];
  //   }
  //   // 可选地，将下三角部分（对角线之下）的元素设为零
  //   for (int i = j + 1; i < SIZE; ++i) { // 遍历对角线之下的行
  //     H_L[i + j * SIZE] = 0.0;
  //   }
  // }

  // std::cout << "the L is :" << std::endl;
  // print_matrix_colmajor(H_L, N, N);
  CHECK_Runtime(
      cudaMemcpy(L, H_L, sizeof(float) * N * N, cudaMemcpyHostToDevice));
  // 上来就算分母A的L2范数
  CHECK_Cublas(cublasSnrm2(handle, N * N, temp_origin, 1, &dresult));
  std::cout << "the dresult is:" << dresult << std::endl;
  // 在syrk之前查看一下temp_origin的模样
  // std::cout << "before syrk the temp_origin is:" << std::endl;
  float *lookthetemporigin;
  lookthetemporigin = (float *)malloc(sizeof(float) * N * N);
  CHECK_Runtime(cudaMemcpy(lookthetemporigin, temp_origin,
                           sizeof(float) * N * N, cudaMemcpyDeviceToHost));

  // print_matrix_colmajor(lookthetemporigin, N, N);
  // 用syrk算出A-LL^T
  CHECK_Cublas(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, N,
                           &alpha, L, N, &beta, temp_origin, N));
  // syrk算出来的结果是对称的，所以syrk只会按照uplo的指示将结果存于下三角或者是上三角，才意识到，所以还需要对其进行一个映射操作，之后再进行nrm2运算

  // std::cout << "after syrk the temp_origin is:" << std::endl;
  CHECK_Runtime(cudaMemcpy(lookthetemporigin, temp_origin,
                           sizeof(float) * N * N, cudaMemcpyDeviceToHost));
  // print_matrix_colmajor(lookthetemporigin, N, N);

  // 将下三角映射到上三角，从而构成一个对称矩阵
  // std::cout << "映射情况：" << std::endl;
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      lookthetemporigin[j * N + i] =
          lookthetemporigin[i * N + j]; // 将下三角的值赋给上三角
    }
  }
  // print_matrix_colmajor(lookthetemporigin, N, N);
  float *fenzi;
  CHECK_Runtime(cudaMalloc((void **)&(fenzi), sizeof(float) * N * N));
  CHECK_Runtime(cudaMemcpy(fenzi, lookthetemporigin, sizeof(float) * N * N,
                           cudaMemcpyHostToDevice));

  // 用Dnrm2算分子的L2范数
  CHECK_Cublas(cublasSnrm2(handle, N * N, fenzi, 1, &nresult));
  std::cout << std::fixed << std::setprecision(20);
  std::cout << "the nresult is:" << nresult << std::endl;

  *result = nresult / dresult;

  CHECK_Runtime(cudaFree(L));
  CHECK_Runtime(cudaFree(temp));
  CHECK_Runtime(cudaFree(temp_origin));
  free(H_L);
  free(H_matrix);

  return EXIT_SUCCESS;
}

int print_matrix_rowmajor(double *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[i * m + j]);
      std::cout << " " << matrix[j * m + i];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

template <typename T> int print_matrix_colmajor(T *matrix, int m, int n) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[j * m + i];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}
