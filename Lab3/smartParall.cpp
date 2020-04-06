#include <cstdlib>
#include <mpi.h>

#define M 4
#define N 5
#define K 5
#define MAX_DIMS 2
// |------x
// |
// |
// y
void fillMatrices(double *a, double *b);
void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D);
void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeC, int sizeRowStrip, int sizeColumnStrip);
void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows);
void fillDataForEachProc(int *dims, int *coords, int *sizeRows, int *sizeCols, int sizeRowStripMod,
                         int sizeColumnStripMod);
void fillScatterAData(int *dims, int **sendCountsA, int **displsA, int sizeRowStripMod);
void fillScatterBData(int *dims, int **sendCountsB, int **displsB);
void fillGathervCData(int *dims, int **sendCountsC, int **displsC, int sizeRowStrip, int sizeComm2);

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
  printf("%d",dims[1]);
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
  /*if (rank == 0) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        printf("%f ", C[i * K + j]);
      }
      printf("\n");
    }
  }*/
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
      a[i * N + j] = i * N + j;
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {

      b[i * K + j] = i * K + j;
    }
  }
}

void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D) {
  MPI_Datatype typeB, typeC;
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
  //cords[0]-y
  //cords[1]-x
  MPI_Comm_rank(comm2D, &rank2Comm);

  fillDataForEachProc(dims, coords, &sizeRowsForEachProc, &sizeColsForEachProc, sizeRowStripMod, sizeColumnStripMod);
  double *aPart = (double *) calloc(sizeRowsForEachProc * N, sizeof(double));
  double *bPart = (double *) calloc(sizeColsForEachProc * N, sizeof(double));
  double *cPart = (double *) calloc(sizeColsForEachProc * sizeRowsForEachProc, sizeof(double));

  createsTypes(&typeB, &typeC, sizeRowsForEachProc, sizeColsForEachProc);

  fillScatterAData(dims, &sendCountsA, &displsA, sizeRowStripMod);
  fillScatterBData(dims, &sendCountsB, &displsB);
  fillGathervCData(dims, &sendCountsC, &displsC, sizeRowStrip, sizeComm2);

  MPI_Comm comm1DColumns;
  MPI_Comm comm1DRows;
  createComms(comm2D, &comm1DColumns, &comm1DRows);

  if (coords[1] == 0) {
    MPI_Scatterv(a, sendCountsA, displsA, MPI_DOUBLE, aPart,
                 sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);
    /*printf("rank=%d recv=%d\n", rank, sizeRowsForEachProc * N);
    for (int i = 0; i < sizeRowsForEachProc * N; ++i) {
      printf("%f ", aPart[i]);
    }
    printf("\n");*/
  }

  if (coords[0] == 0) {
    for (int j = 0; j < dims[1]; ++j) {
      printf("send=%d displs=%d\n",sendCountsB[j],displsB[j]);

    } printf("\n");
    MPI_Scatterv(b, sendCountsB, displsB, typeB, bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);

     printf("rank=%d ", rank);

     for (int i = 0; i < sizeColsForEachProc*N; ++i) {
       printf("%f ", bPart[i]);
     }
     printf("\n");
  }

  //MPI_Bcast(aPart, sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);
  /*if(coords[1]==1 && coords[0]==1)
  for (int i = 0; i < sizeRowsForEachProc*N; ++i) {
    printf("%f ", aPart[i]);
  }
  printf("\n");*/
  //MPI_Bcast(bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);
  /* if(coords[1]==1 && coords[0]==1)
     for (int i = 0; i < sizeColsForEachProc*N; ++i) {
       printf("%f ", bPart[i]);
     }*/

  /*for (int i = 0; i < sizeRowsForEachProc; ++i) {
    for (int j = 0; j < sizeColsForEachProc; ++j) {
      for (int k = 0; k < N; ++k) {
        cPart[i * sizeColsForEachProc + j] += aPart[i * N + k] * bPart[k * sizeColsForEachProc + j];
        *//*printf("c[%d]=%f\t  a[%d]=%f\t b[%d]=%f\n",i * sizeColsForEachProc + j,cPart[i * sizeColsForEachProc + j],
            i * N + k,aPart[i * N + k],
        k * sizeRowsForEachProc + j,bPart[k * sizeColsForEachProc + j] );*//*
      }
    }
  }*/

  /*for (int i = 0; i < sizeRowsForEachProc; ++i) {
    for (int j = 0; j < sizeColsForEachProc; ++j) {
      printf("%f ",cPart[i * sizeColsForEachProc + j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("\n");*/

  //MPI_Gatherv(cPart, sizeColumnStrip * sizeRowStrip, MPI_DOUBLE, c, sendCountsC, displsC, typeC, 0, comm2D);

  /*if (dims[1] - 1 == coords[1])
    MPI_Send(cPart, sizeColsForEachProc * sizeRowsForEachProc, MPI_DOUBLE, 0, 0, comm2D);
  if (coords[1] == 0 && coords[0] == 0) {
   // for (int i = 0; i < dims[1]; ++i) {
      MPI_Recv(c + (K / sizeColumnStrip - 1) * sizeColumnStrip, 1, typeCModCol, i, 0, comm2D, MPI_STATUS_IGNORE);
   // }

  }*/

  /*for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      printf("%f ", cPart1[i * M + j]);
    }
    printf("\n");
  }*/

  if (rank == 0) {
    free(sendCountsB);
    free(displsB);
    free(sendCountsC);
    free(displsC);
    free(sendCountsA);
    free(displsA);
    MPI_Type_free(&typeB);
    MPI_Type_free(&typeC);
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

void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeC, int sizeRowStrip, int sizeColumnStrip) {

  MPI_Type_vector(N, sizeColumnStrip, K, MPI_DOUBLE, typeB);
  MPI_Type_vector(sizeRowStrip, sizeColumnStrip, K, MPI_DOUBLE, typeC);

  MPI_Type_create_resized(*typeB, 0, sizeColumnStrip * sizeof(double), typeB);
  MPI_Type_create_resized(*typeC, 0, sizeColumnStrip * sizeof(double), typeC);

  MPI_Type_commit(typeB);
  MPI_Type_commit(typeC);
}

void fillDataForEachProc(int *dims, int *coords, int *sizeRows, int *sizeCols, int sizeRowStripMod,
                         int sizeColumnStripMod) {
  //FillSizeRowsForEachProc
  if (coords[0] < sizeRowStripMod) {
    *sizeRows = M / dims[0] + 1;
  } else {
    *sizeRows = M / dims[0];
  }
  //FillSizeColsForEachProc
  if (coords[1] < sizeColumnStripMod) {
    *sizeCols = K / dims[1] + 1;
  } else {
    *sizeCols = K / dims[1];
  }
  printf("coords[0]=%d coords[1]=%d \n", coords[0], coords[1]);
  printf("sizeRows=%d sizeCols=%d \n", *sizeRows, *sizeCols);
}

void fillScatterAData(int *dims, int **sendCountsA, int **displsA, int sizeRowStripMod) {

  *sendCountsA = (int *) calloc(dims[0], sizeof(int));
  *displsA = (int *) calloc(dims[0], sizeof(int));
  for (int i = 0; i < dims[0]; ++i) {
    if (i < sizeRowStripMod) {
      (*displsA)[i] = (M / dims[0] + 1) * i * N;
      (*sendCountsA)[i] = (M / dims[0] + 1) * N;
    } else {
      (*displsA)[i] = N * ((M / dims[0] + 1) * sizeRowStripMod + (i - sizeRowStripMod) * (M / dims[0]));
      (*sendCountsA)[i] = (M / dims[0]) * N;
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
