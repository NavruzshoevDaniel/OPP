#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <mpi.h>

/*#define sizeMatrix 15000
#define epsilon 0.0000000001*/
#define sizeMatrix 25000
#define epsilon 0.00001
//скалярное произведение
double scalarMultiply(double *vec1, double *vec2) {

  double result = 0;

  for (int i = 0; i < sizeMatrix; i++)
    result += vec1[i] * vec2[i];

  return result;
}

//норма
double norm(double *vectorPart, int numberOfElements) {

  double result_part = 0;
  double result = 0;
  for (int i = 0; i < numberOfElements; ++i) {
    result_part += vectorPart[i] * vectorPart[i];
  }

  MPI_Allreduce(&result_part, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sqrt(result);
}

void fillTestDataMPI(double *matrixMpi, double *b, double *x, int shift, int numberOfElem) {

  for (int i = 0; i < numberOfElem; i++) {
    for (int j = 0; j < sizeMatrix; j++) {
      matrixMpi[i * sizeMatrix + j] = 1.0;
      if (shift + i == j)
        ++matrixMpi[i * sizeMatrix + j];
    }
    b[i] = sizeMatrix + 1;
    x[i] = 0;
  }

}

void initRAndZ(double *rPart,
               double *bPart,
               double *zPart,
               double *APart,
               double *xPart,
               int *numberOfElements,
               int *shift,
               int rank,
               int size) {
  double *aXPart = (double *) calloc(numberOfElements[rank], sizeof(double));
  int currentRank = (rank + size) % size;
  for (int i = 0; i < size; ++i) {
    for (int position = 0; position < numberOfElements[rank]; ++position) {
      for (int j = shift[currentRank]; j < shift[currentRank] + numberOfElements[currentRank]; ++j) {
        aXPart[position] += APart[j + position * sizeMatrix] * xPart[j - shift[currentRank]];
      }
    }
    currentRank = (rank - i - 1 + size) % size;

    MPI_Sendrecv_replace(xPart,
                         numberOfElements[0],
                         MPI_DOUBLE,
                         (rank + 1) % size,
                         0,
                         (rank - 1 + size) % size,
                         0,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < numberOfElements[rank]; ++i) {
    rPart[i] = bPart[i] - aXPart[i];
    zPart[i] = rPart[i];
  }
  free(aXPart);
}

double calculateAlpha(double *rPart, double *zPart, double *aZPart, int numberOfElements, int shift) {

  double sTop = 0, sumTop;//числитель
  double sDown = 0, sumDown;//знаменатель

  for (int i = 0; i < numberOfElements; ++i) {
    sTop += rPart[i] * rPart[i];
    sDown += aZPart[i + shift] * zPart[i];
  }
  MPI_Allreduce(&sTop, &sumTop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sDown, &sumDown, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sumTop / sumDown;
}

void calculateAZPart(double *APart, double *aZPart, double *z, int numberOfElements, int shift) {
  for (int i = 0; i < numberOfElements; ++i) {
    aZPart[i + shift] = scalarMultiply(&APart[i * sizeMatrix], z);
  }

}

int main(int argc, char *argv[]) {
  int size, rank;

  MPI_Init(&argc, &argv);//инициализация mpi
  MPI_Comm_size(MPI_COMM_WORLD, &size);//получение числа процессов
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);//получение номера процесса

  double startTime = MPI_Wtime();

  int *numberOfElements = (int *) calloc(size, sizeof(int));//количество элементов в процессе

  int *shift = (int *) calloc(size, sizeof(int));//сдвиг

  for (int i = 0; i < size; ++i)
    numberOfElements[i] = (sizeMatrix / size) + ((i < sizeMatrix % size) ? (1) : (0));

  for (int i = 1; i < size; ++i)
    shift[i] = shift[i - 1] + numberOfElements[i - 1];

  double *b_MPI = (double *) calloc(numberOfElements[rank], sizeof(double));//части b-вектора
  double *x_MPI = (double *) calloc(numberOfElements[0], sizeof(double));//части x-вектора
  double *A_MPI = (double *) calloc(sizeMatrix * numberOfElements[rank], sizeof(double));//части A-матрицы
  double *prevR = (double *) calloc(numberOfElements[rank], sizeof(double));//части rn-1-вектора
  double *r_MPI = (double *) calloc(numberOfElements[rank], sizeof(double));//части r-вектора
  double *z_MPI = (double *) calloc(numberOfElements[rank], sizeof(double));//части z-вектора


  fillTestDataMPI(A_MPI, b_MPI, x_MPI, shift[rank], numberOfElements[rank]);
  //r и z

  initRAndZ(r_MPI, b_MPI, z_MPI, A_MPI, x_MPI, numberOfElements, shift, rank, size);

  double z[sizeMatrix];
  MPI_Allgatherv(z_MPI, numberOfElements[rank], MPI_DOUBLE, z, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);

  //норма b
  double bNorm = norm(b_MPI, numberOfElements[rank]);

  double alpha = 0, betta = 0;
  double sTop = 0, sumTop;//числитель
  double sDown = 0, sumDown;//знаменатель

  double *aZ_part = (double *) calloc(sizeMatrix, sizeof(double));//A*z

  double condition = 100;
  while (condition > epsilon) {

    calculateAZPart(A_MPI, aZ_part, z, numberOfElements[rank], shift[rank]);
    alpha = calculateAlpha(r_MPI, z_MPI, aZ_part, numberOfElements[rank], shift[rank]);

    //считаем xn+1
    for (int i = 0; i < numberOfElements[rank]; ++i) {
      x_MPI[i] += alpha * z_MPI[i];
    }

    //запоминаем rn
    for (int i = 0; i < numberOfElements[rank]; ++i) {
      prevR[i] = r_MPI[i];
    }

    //считаем rn+1
    for (int i = 0; i < numberOfElements[rank]; ++i) {
      r_MPI[i] += -1 * alpha * aZ_part[i + shift[rank]];
    }

    //считаем betta
    for (int i = 0; i < numberOfElements[rank]; ++i) {
      sDown += prevR[i] * prevR[i];
      sTop += r_MPI[i] * r_MPI[i];
    }
    MPI_Allreduce(&sTop, &sumTop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sDown, &sumDown, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    betta = sumTop / sumDown;

    //получаем след zn+1
    for (int i = 0; i < numberOfElements[rank]; ++i) {
      z_MPI[i] = r_MPI[i] + betta * z_MPI[i];
    }
    condition = sqrt(sumTop) / bNorm;

  }

/*  for (int i = 0; i < numberOfElements[rank]; ++i) {
    printf("%f ", x_MPI[i]);
  }*/

  free(A_MPI);
  free(r_MPI);
  free(z_MPI);
  free(aZ_part);
  free(numberOfElements);
  free(shift);
  free(x_MPI);
  free(b_MPI);
  free(prevR);

  if (rank == 0)
    printf("Execution time %lf \n", MPI_Wtime() - startTime);

  MPI_Finalize();
  return 0;
}
