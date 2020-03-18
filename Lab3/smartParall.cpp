#include <cstdlib>
#include <mpi.h>

#define M 16
#define N 4
#define K 17
#define MAX_DIMS 2

void fillMatrices(double *a, double *b);
void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D);
void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeBMod, MPI_Datatype *typeC, MPI_Datatype *typeCModRow,
                  MPI_Datatype *typeCModCol, MPI_Datatype *typeCModRowCol,
                  int sizeRowStrip, int sizeColumnStrip, int sizeColumnStripMod,
                  int sizeRowStripMod);
void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows);
void fillDataForEachProc(int *dims, int *coords, int *sizeRows, int *sizeCols);
void fillScatterAData(int *dims, int **sendCountsA, int **displsA, int sizeRowStrip, int sizeRowStripMod);
void fillScatterBData(int *dims, int **sendCountsB, int **displsB);
void fillGathervCData(int *dims, int **sendCountsC, int **displsC, int sizeRowStrip,int sizeComm2);

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
  if (rank == 0) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        printf("%f ", C[i * M + j]);
      }
      printf("\n");
    }
  }
  if (rank == 0) {
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
  MPI_Datatype typeB, typeBMod, typeC, typeCModRow, typeCModCol,typeCModRowCol;
  int *sendCountsB = NULL, *displsB = NULL;
  int *sendCountsA = NULL, *displsA = NULL;
  int *sendCountsC = NULL, *displsC = NULL;

  int coords[2];

  int sizeComm2;
  int rank2Comm;

  int sizeRowStrip = M / dims[0];
  int sizeRowStripMod = M % dims[0];
  int sizeColumnStrip = K / dims[1];
  int sizeColumnStripMod = K % dims[1];
  int sizeRowsForEachProc, sizeColsForEachProc;

  MPI_Comm_size(comm2D, &sizeComm2);
  MPI_Cart_coords(comm2D, rank, MAX_DIMS, coords);
  MPI_Comm_rank(comm2D, &rank2Comm);

  fillDataForEachProc(dims, coords, &sizeRowsForEachProc, &sizeColsForEachProc);

  double *aPart = (double *) calloc(sizeRowsForEachProc * N, sizeof(double));
  double *bPart = (double *) calloc(sizeColsForEachProc * N, sizeof(double));
  double *cPart = (double *) calloc(sizeColsForEachProc * sizeRowsForEachProc, sizeof(double));
  double *cPart1 = (double *) calloc(M * K, sizeof(double));

  if (rank == 0) {
    createsTypes(&typeB, &typeBMod, &typeC, &typeCModRow, &typeCModCol,&typeCModRowCol,
                 sizeRowStrip, sizeColumnStrip, sizeColumnStripMod, sizeRowStripMod);

    fillScatterAData(dims, &sendCountsA, &displsA, sizeRowStrip, sizeRowStripMod);
    fillScatterBData(dims, &sendCountsB, &displsB);
    fillGathervCData(dims, &sendCountsC, &displsC, sizeRowStrip,sizeComm2);
  }

  MPI_Comm comm1DColumns;
  MPI_Comm comm1DRows;
  createComms(comm2D, &comm1DColumns, &comm1DRows);

  if (coords[1] == 0) {
    MPI_Scatterv(a, sendCountsA, displsA, MPI_DOUBLE, aPart, sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);
    /*printf("rank=%d recv=%d\n", rank, sizeRowsForEachProc*N);
    for (int i = 0; i < sizeRowsForEachProc*N; ++i) {
      printf("%f ", aPart[i]);
    }*/
  }

  if (coords[0] == 0) {

    MPI_Scatterv(b, sendCountsB, displsB, typeB, bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);

    if (coords[1] == 0) {
      MPI_Send(b + (K / sizeColumnStrip - 1) * sizeColumnStrip, 1, typeBMod, dims[1] - 1, 0, comm1DRows);
    }
    if (dims[1] - 1 == coords[1])
      MPI_Recv(bPart, sizeColumnStrip * N + sizeColumnStripMod * N, MPI_DOUBLE, 0, 0, comm1DRows, MPI_STATUS_IGNORE);

    /* printf("rank=%d ", rank);

     for (int i = 0; i < sizeColsForEachProc*N; ++i) {
       printf("%f ", bPart[i]);
     }
     printf("\n");*/
  }

  MPI_Bcast(aPart, sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);
  /*if(coords[1]==1 && coords[0]==1)
  for (int i = 0; i < sizeRowsForEachProc*N; ++i) {
    printf("%f ", aPart[i]);
  }
  printf("\n");*/
  MPI_Bcast(bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);
  /* if(coords[1]==1 && coords[0]==1)
     for (int i = 0; i < sizeColsForEachProc*N; ++i) {
       printf("%f ", bPart[i]);
     }*/

  for (int i = 0; i < sizeRowsForEachProc; ++i) {
    for (int j = 0; j < sizeColsForEachProc; ++j) {
      for (int k = 0; k < N; ++k) {
        cPart[i * sizeColsForEachProc + j] += aPart[i * N + k] * bPart[k * sizeColsForEachProc + j];
        /*printf("c[%d]=%f\t  a[%d]=%f\t b[%d]=%f\n",i * sizeColsForEachProc + j,cPart[i * sizeColsForEachProc + j],
            i * N + k,aPart[i * N + k],
        k * sizeRowsForEachProc + j,bPart[k * sizeColsForEachProc + j] );*/
      }
    }
  }

  /*for (int i = 0; i < sizeRowsForEachProc; ++i) {
    for (int j = 0; j < sizeColsForEachProc; ++j) {
      printf("%f ",cPart[i * sizeColsForEachProc + j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("\n");*/

  MPI_Gatherv(cPart, sizeColumnStrip * sizeRowStrip, MPI_DOUBLE, c, sendCountsC, displsC, typeC, 0, comm2D);

  /*if (dims[1] - 1 == coords[1])
    MPI_Send(cPart, sizeColsForEachProc * sizeRowsForEachProc, MPI_DOUBLE, 0, 0, comm2D);
  if (coords[1] == 0 && coords[0] == 0) {
   // for (int i = 0; i < dims[1]; ++i) {
      MPI_Recv(c + (K / sizeColumnStrip - 1) * sizeColumnStrip, 1, typeCModCol, i, 0, comm2D, MPI_STATUS_IGNORE);
   // }

  }*/

  if (dims[1] - 1 == coords[1] && coords[0] != dims[0] - 1) {
     printf("%d %d\n", coords[0], rank2Comm);
    MPI_Send(cPart, sizeColsForEachProc * sizeRowsForEachProc, MPI_DOUBLE, 0, 0, comm2D);
  }
  if (dims[0] - 1 == coords[0] && coords[1] != dims[1] - 1) {
    printf("==%d %d\n", coords[1], rank2Comm);
    MPI_Send(cPart, sizeColsForEachProc * sizeRowsForEachProc, MPI_DOUBLE, 0, 0, comm2D);
  }
  if (coords[1] == 0 && coords[0] == 0) {
    int recvRankCol = dims[0]-1;
    int recvRankRow = dims[0]*(dims[1]-1);
    for (int i = 0; i < dims[0] - 1; ++i) {
      printf("%d \n",recvRankCol);
      printf("myval=%d \n",i * sizeColumnStrip * K + (K / sizeColumnStrip - 1) * sizeColumnStrip);
      MPI_Recv(cPart1 + i * sizeColumnStrip * K + (K / sizeColumnStrip - 1) * sizeColumnStrip, 1,
               typeCModCol, recvRankCol, 0, comm2D, MPI_STATUS_IGNORE);
      recvRankCol += dims[1];
    }

    for (int i = 0; i < dims[1] - 1; ++i) {
      printf("==%d \n",recvRankRow);
      MPI_Recv(c + i*sizeRowStrip+K * (N / sizeRowStrip - 1) * sizeRowStrip,
               1, typeCModRow, recvRankRow, 0, comm2D, MPI_STATUS_IGNORE);
      recvRankRow += 1;
    }

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        printf("%f ",cPart1[i*M+j]);
      }
      printf("\n");
    }
    printf("\n");printf("\n");
    /*for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        printf("%f ", cPart1[i * M + j]);
      }
      printf("\n");
    }*/
  }

    if (rank == 0) {
      free(sendCountsB);
      free(displsB);
      free(sendCountsC);
      free(displsC);
      free(sendCountsA);
      free(displsA);
      MPI_Type_free(&typeB);
      MPI_Type_free(&typeC);
      MPI_Type_free(&typeCModCol);
    }
    free(aPart);
    free(bPart);
    free(cPart);
  }

  void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows) {
    int remainsRow[2] = {0, 1};
    int remainsColumns[2] = {1, 0};

    MPI_Cart_sub(comm2D, remainsColumns, columns);
    MPI_Cart_sub(comm2D, remainsRow, rows);
  }

  void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeBMod, MPI_Datatype *typeC, MPI_Datatype *typeCModRow,
                    MPI_Datatype *typeCModCol, MPI_Datatype *typeCModRowCol,
                    int sizeRowStrip, int sizeColumnStrip, int sizeColumnStripMod,
                    int sizeRowStripMod) {

    MPI_Type_vector(N, sizeColumnStrip, K, MPI_DOUBLE, typeB);
    MPI_Type_vector(N, sizeColumnStrip + sizeColumnStripMod, K, MPI_DOUBLE, typeBMod);
    MPI_Type_vector(sizeRowStrip, sizeColumnStrip, K, MPI_DOUBLE, typeC);
    MPI_Type_vector(sizeRowStrip, sizeColumnStrip + sizeColumnStripMod, K, MPI_DOUBLE, typeCModCol);
    MPI_Type_vector(sizeRowStrip+sizeRowStripMod, sizeColumnStrip, K, MPI_DOUBLE, typeCModRow);
    MPI_Type_vector(sizeRowStrip+sizeRowStripMod, sizeColumnStrip + sizeColumnStripMod, K, MPI_DOUBLE, typeCModRowCol);

    MPI_Type_create_resized(*typeB, 0, sizeColumnStrip * sizeof(double), typeB);
    MPI_Type_create_resized(*typeC, 0, sizeColumnStrip * sizeof(double), typeC);

    MPI_Type_commit(typeB); //регистрируем новый производный тип
    MPI_Type_commit(typeBMod); //регистрируем новый производный тип
    MPI_Type_commit(typeC); //регистрируем новый производный тип
    MPI_Type_commit(typeCModCol); //регистрируем новый производный тип
    MPI_Type_commit(typeCModRow); //регистрируем новый производный тип
    MPI_Type_commit(typeCModRowCol); //регистрируем новый производный тип
  }

  void fillDataForEachProc(int *dims, int *coords, int *sizeRows, int *sizeCols) {
    //FillSizeRowsForEachProc
    if (coords[0] == dims[0] - 1) {
      *sizeRows = M / dims[0] + M % dims[0];
    } else {
      *sizeRows = M / dims[0];
    }
    //FillSizeColsForEachProc
    if (coords[1] == dims[1] - 1) {
      *sizeCols = K / dims[1] + K % dims[1];
    } else {
      *sizeCols = K / dims[1];
    }
  }

  void fillScatterAData(int *dims, int **sendCountsA, int **displsA, int sizeRowStrip, int sizeRowStripMod) {

    *sendCountsA = (int *) calloc(dims[0], sizeof(int));
    *displsA = (int *) calloc(dims[0], sizeof(int));
    for (int i = 0; i < dims[0]; ++i) {
      if (dims[0] - 1 == i) {
        (*displsA)[i] = i * sizeRowStrip * N;
        (*sendCountsA)[i] = (sizeRowStrip + sizeRowStripMod) * N;
      } else {
        (*displsA)[i] = i * sizeRowStrip * N;
        (*sendCountsA)[i] = sizeRowStrip * N;
      }
    }
  }

  void fillScatterBData(int *dims, int **sendCountsB, int **displsB) {
    *sendCountsB = (int *) calloc(dims[1], sizeof(int));
    *displsB = (int *) calloc(dims[1], sizeof(int));

    for (int i = 0; i < dims[1]; ++i) {
      (*displsB)[i] = i;
      (*sendCountsB)[i] = 1;
    }
  }

  void fillGathervCData(int *dims, int **sendCountsC, int **displsC, int sizeRowStrip,
                        int sizeComm2) {
    *sendCountsC = (int *) calloc(sizeComm2, sizeof(int));
    *displsC = (int *) calloc(sizeComm2, sizeof(int));
    for (int i = 0; i < sizeComm2; ++i) {
      (*sendCountsC)[i] = 1;
    }
    for (int i = 0; i < dims[0]; ++i) {
      for (int j = 0; j < dims[1]; ++j) {
          (*displsC)[i * dims[1] + j] = i * dims[1] * sizeRowStrip + j;
      }
    }
  }
