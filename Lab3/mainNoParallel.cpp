//
// Created by Daniel on 08.03.2020.
//
#include <cstdio>
#include <cstdlib>
#define M 4
#define N 4
#define K 4


void matrixMultiply(double *a, double *b, double *result);
void fillMatrices(double *a, double *b);

int main(int argc, char *argv[]) {

  double *A = (double *) calloc(M * N, sizeof(double));
  double *B = (double *) calloc(K * N, sizeof(double));
  double *res = (double *) calloc(K * M, sizeof(double));
  fillMatrices(A, B);
  matrixMultiply(A, B, res);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      printf("%f ",res[i*K+j]);
    }
    printf("\n");
  }
}

void matrixMultiply(double *a, double *b, double *result) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        result[i * K + k] += a[i * N + j] * b[j * K + k];
      }
    }
  }

}

void fillMatrices(double *a, double *b) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i * N + j] = i * N + j;
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      b[i * K + j] = i * K + j;
    }
  }

}