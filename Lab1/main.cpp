#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <mpi.h>

#define sizeMatrix 5
#define epsilon 0.0000000000000000000000000000000001

//перемножение матрицы на вектор
void matrixVectorMultiply(double *matrix, double *vector, double *result) {

  for (int i = 0; i < sizeMatrix; i++){
    result[i]=0;
    for (int j = 0; j < sizeMatrix; j++){
      result[i] += matrix[i * sizeMatrix + j] * vector[j];
    }
  }

}

//сложение векторов
void sumVectors(double *vec1, double *vec2, double *result) {

  for (int i = 0; i < sizeMatrix; i++) {
    result[i] = vec1[i] + vec2[i];
  }

}

//скалярное произведение
double scalarMultiply(double *vec1, double *vec2) {

  double result = 0;

  for (int i = 0; i < sizeMatrix; i++)
    result += vec1[i] * vec2[i];

  return result;
}

//норма
double norm(double *vector) {

  double result = 0;

  for (int i = 0; i < sizeMatrix; i++)
    result += vector[i] * vector[i];

  return sqrt(result);
}

void initRAndZ(double *A, double *x, double *b, double *r, double *z) {
  double tmp = 0;
  for (int i = 0; i < sizeMatrix; ++i) {
    for (int j = 0; j < sizeMatrix; ++j) {
      tmp += A[i * sizeMatrix + j] * x[j];
    }
    r[i] = b[i] - tmp;
    z[i] = r[i];
    tmp = 0;
  }

}

void fillTestData(double *matrix, double *b, double *x) {
  for (int i = 0; i < sizeMatrix; i++) {
    for (int j = 0; j < sizeMatrix; j++) {
      if (i == j) {
        matrix[i * sizeMatrix + j] = 2.0;
      } else {
        matrix[i * sizeMatrix + j] = 1.0;
      }
    }
    b[i] = sizeMatrix + 1;
    x[i] = 0;
  }
}


void secondTestData(double *matrix, double *b, double *x,double *u) {
  printf("Init b value:");
  for (int i = 0; i < sizeMatrix; i++) {
    for (int j = 0; j < sizeMatrix; j++) {
      if (i == j) {
        matrix[i * sizeMatrix + j] = 2.0f;
      } else {
        matrix[i * sizeMatrix + j] = 1.0f;
      }
    }

    u[i] = sin(2 * 3.14 * i / sizeMatrix);
    //printf("%f ",u[i]);

    x[i]=0;

  }
 // printf("\n");
  matrixVectorMultiply(matrix,u,b);
 // printf("\n");
}

double conditional(double normR, double normB) {
  return normR / normB;
}

double calculateAlpha(double *A, double *r, double *z, double *aZPart) {

  matrixVectorMultiply(A, z, aZPart);
  double sTop = 0, sumTop;//числитель
  double sDown = 0, sumDown;//знаменатель

  for (int i = 0; i < sizeMatrix; ++i) {
    sTop += r[i] * r[i];
    sDown += aZPart[i] * z[i];
  }

  return sTop / sDown;
}

double calculateBettaAndNormR(double *r, double *prevR, double *normR) {
  double tmp = scalarMultiply(r, r);
  *normR = sqrt(tmp);
  return tmp / scalarMultiply(prevR, prevR);

}

void nextX(double *z, double alpha, double *x) {
  for (int i = 0; i < sizeMatrix; ++i) {
    x[i] += alpha * z[i];
  }
}

void nextR(double *A, double alpha, double *aZPart, double *r) {
  for (int i = 0; i < sizeMatrix; ++i) {
    r[i] += -1 * alpha * aZPart[i];
  }
}

void nextZ(double *z, double betta, double *r) {
  for (int i = 0; i < sizeMatrix; ++i) {
    z[i] = r[i] + betta * z[i];
  }
}
void printVect(double*vec){
  for (int i = 0; i < sizeMatrix; ++i) {
    printf("%f ",vec[i]);
  }
  printf("%\n");
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);//инициализация mpi
  double startTime = MPI_Wtime();
  double *A = (double *) calloc(sizeMatrix * sizeMatrix, sizeof(double));
  double *b = (double *) calloc(sizeMatrix, sizeof(double));
  double *prevR = (double *) calloc(sizeMatrix, sizeof(double));
  double *x = (double *) calloc(sizeMatrix, sizeof(double));
  double *r = (double *) calloc(sizeMatrix, sizeof(double));
  double *z = (double *) calloc(sizeMatrix, sizeof(double));
  double *u = (double *) calloc(sizeMatrix, sizeof(double));

  secondTestData(A,b,x,u);
  //fillTestData(A,b,x);

  initRAndZ(A, x, b, r, z);

  double alpha = 0;
  double betta = 0;
  double normB = norm(b);
  double normR = norm(r);
  double *aZ_part = (double *) calloc(sizeMatrix, sizeof(double));//A*z
  while (conditional(normR, normB) > epsilon) {

    alpha = calculateAlpha(A, r, z, aZ_part);

    nextX(z, alpha, x);//Xn+1

    for (int i = 0; i < sizeMatrix; i++)
      prevR[i] = r[i];

    nextR(A, alpha, aZ_part, r);//Rn+1

    betta = calculateBettaAndNormR(r, prevR, &normR);

    nextZ(z, betta, r);//Zn+1
  }

  /*for (int i = 0; i < sizeMatrix; i++) {
    printf("%f ", x[i]);
  }*/

  printf("Execution time %lf \n", MPI_Wtime() - startTime);
  free(A);
  free(b);
  free(r);
  free(x);
  free(z);
  free(u);
  MPI_Finalize();
  return 0;
}
