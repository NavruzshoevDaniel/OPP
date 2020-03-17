#include <cstdlib>
#include <mpi.h>

#define M 4
#define N 4
#define K 4
#define MAX_DIMS 2

void fillMatrices(double *a, double *b);
void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D);
void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeC, int sizeRowStrip, int sizeColumnStrip);
void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows);

int main(int argc, char *argv[]) {
  int size, rank;
  MPI_Init(&argc, &argv);//инициализация mpi
  MPI_Comm_size(MPI_COMM_WORLD, &size);//получение числа процессов
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);//получение номера процесса
  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  int reorder = 0;
  MPI_Comm comm2D;
  MPI_Dims_create(size, MAX_DIMS, dims);
  MPI_Cart_create(MPI_COMM_WORLD, MAX_DIMS, dims, periods, reorder, &comm2D);

  double *A;
  double *B;
  double *C;

  if (rank == 0) {
    A = (double *) calloc(M * N, sizeof(double));
    B = (double *) calloc(K * N, sizeof(double));
    C = (double *) calloc(K * M, sizeof(double));
    fillMatrices(A, B);
  }

  caluclate(A, B, C, dims, rank, comm2D);
  if(rank==0){
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        printf("%f ",C[i * M + j]);
      }
      printf("\n");
    }
  }
  if(rank==0){
    free(A);
    free(B);
    free(C);

  }
  MPI_Finalize();
  return 0;
}

void fillMatrices(double *a, double *b) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i * N + j] = 1;
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {

      b[i * K + j] = 1;
    }
  }
}

void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D) {
  MPI_Datatype typeB, typeC;
  int *sendCountsB = NULL, *displsB = NULL;
  int *sendCountsC = NULL, *displsC = NULL;

  int coords[2];

  int sizeComm2;
  int sizeRowStrip = M / dims[0];
  int sizeColumnStrip = K / dims[1];

  MPI_Comm_size(comm2D, &sizeComm2);
  MPI_Cart_coords(comm2D, rank, MAX_DIMS, coords);

  double *aPart = (double *) calloc(sizeRowStrip * N, sizeof(double));
  double *bPart = (double *) calloc(sizeColumnStrip * N, sizeof(double));
  double *cPart = (double *) calloc(sizeColumnStrip * sizeRowStrip, sizeof(double));

  if (rank == 0) {
    createsTypes(&typeB, &typeC, sizeRowStrip, sizeColumnStrip);
    sendCountsB = (int *) calloc(dims[1], sizeof(int));
    displsB = (int *) calloc(dims[1], sizeof(int));
    for (int i = 0; i < dims[1]; ++i) {
      displsB[i] = i;
      sendCountsB[i] = 1;
    }
    sendCountsC = (int *) calloc(sizeComm2, sizeof(int));
    displsC = (int *) calloc(sizeComm2, sizeof(int));
    for (int i = 0; i < sizeComm2; ++i) {
      sendCountsC[i] = 1;
    }
    for (int i = 0; i < dims[0]; ++i) {
      for (int j = 0; j < dims[1]; ++j) {
        displsC[i*dims[1]+j]=i*dims[1]*sizeRowStrip+j;
      }
    }
  }

  MPI_Comm comm1DColumns;
  MPI_Comm comm1DRows;
  createComms(comm2D,&comm1DColumns,&comm1DRows);

  if (coords[1] == 0) {
    MPI_Scatter(a, sizeRowStrip * N, MPI_DOUBLE, aPart, sizeRowStrip * N, MPI_DOUBLE, 0, comm1DColumns);
  }
  if (coords[0] == 0) {
    MPI_Scatterv(b, sendCountsB, displsB, typeB, bPart, sizeColumnStrip * N, MPI_DOUBLE, 0, comm1DRows);
  }

  MPI_Bcast(aPart, sizeRowStrip * N, MPI_DOUBLE, 0, comm1DRows);
  MPI_Bcast(bPart, sizeColumnStrip * N, MPI_DOUBLE, 0, comm1DColumns);

  for (int i = 0; i < sizeRowStrip; ++i) {
    for (int j = 0; j < sizeColumnStrip; ++j) {
      for (int k = 0; k < N; ++k) {
        cPart[i * sizeColumnStrip + j] += aPart[i * N + k] * bPart[k * sizeRowStrip + j];
      }
    }
  }

  MPI_Gatherv(cPart, sizeColumnStrip * sizeRowStrip, MPI_DOUBLE, c, sendCountsC, displsC, typeC, 0, comm2D);
  if (rank == 0) {
    free(sendCountsB);
    free(displsB);
    free(sendCountsC);
    free(displsC);
  }
  free(aPart);
  free(bPart);
  free(cPart);
}
void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows) {
  int remainsRow[2]={0,1};
  int remainsColumns[2]={1,0};

  MPI_Cart_sub(comm2D,remainsColumns,columns);
  MPI_Cart_sub(comm2D,remainsRow,rows);
}

void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeC, int sizeRowStrip, int sizeColumnStrip) {

  MPI_Type_vector(N, sizeColumnStrip, K, MPI_DOUBLE, typeB);
  MPI_Type_vector(sizeRowStrip, sizeColumnStrip, K, MPI_DOUBLE, typeC);

  MPI_Type_create_resized(*typeB, 0, sizeColumnStrip * sizeof(double), typeB);
  MPI_Type_create_resized(*typeC, 0, sizeColumnStrip * sizeof(double), typeC);

  MPI_Type_commit(typeB); //регистрируем новый производный тип
  MPI_Type_commit(typeC); //регистрируем новый производный тип
}
