#include <cstdlib>
#include <mpi.h>

#define M 4
#define N 4
#define K 5
#define MAX_DIMS 2

void fillMatrices(double *a, double *b);
void caluclate(double *a, double *b, double *c, int *dims, int rank, MPI_Comm comm2D);
void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeBMod, int sizeColumnStrip, int sizeColumnStripMod);
void createComms(MPI_Comm comm2D, MPI_Comm *columns, MPI_Comm *rows);
void fillDataForEachProc(int *dims, int *coords, int *sizeRows, int *sizeCols);
void fillScatterAData(int *dims, int **sendCountsA, int **displsA, int sizeRowStrip, int sizeRowStripMod);
void fillScatterBData(int *dims, int **sendCountsB, int **displsB);
void sendRecvMatrix(double *c, double *CPart,int *dims,int *recvCountsY,int splitX,int splitY,int countA, int rank);
void collectCMatrix(int *dims, int *coords, double *c,
                    double *cPart, int sizeRowStrip, int sizeRowStripMod, int sizeColumnStrip,
                    int sizeColumnStripMod,int rank);

int main(int argc, char *argv[]) {
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  int reorder = 0;
  MPI_Comm comm2D;
  MPI_Dims_create(size, MAX_DIMS, dims);
  MPI_Cart_create(MPI_COMM_WORLD, MAX_DIMS, dims, periods, reorder, &comm2D);
  //dims[0]=x
  //dims[1]=y
  /* |--------->y
     |
     |
     x
  */
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
  MPI_Datatype typeB, typeBMod;
  int *sendCountsB = NULL, *displsB = NULL;
  int *sendCountsA = NULL, *displsA = NULL;

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

  if (rank == 0) {
    createsTypes(&typeB, &typeBMod, sizeColumnStrip, sizeColumnStripMod);

    fillScatterAData(dims, &sendCountsA, &displsA, sizeRowStrip, sizeRowStripMod);
    fillScatterBData(dims, &sendCountsB, &displsB);
  }

  MPI_Comm comm1DColumns;
  MPI_Comm comm1DRows;
  createComms(comm2D, &comm1DColumns, &comm1DRows);

  if (coords[1] == 0) {
    MPI_Scatterv(a, sendCountsA, displsA, MPI_DOUBLE, aPart, sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);
  }

  if (coords[0] == 0) {

    MPI_Scatterv(b, sendCountsB, displsB, typeB, bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);

    if (coords[1] == 0) {
      MPI_Send(b + (K / sizeColumnStrip - 1) * sizeColumnStrip, 1, typeBMod, dims[1] - 1, 0, comm1DRows);
    }
    if (dims[1] - 1 == coords[1])
      MPI_Recv(bPart, sizeColumnStrip * N + sizeColumnStripMod * N, MPI_DOUBLE, 0, 0, comm1DRows, MPI_STATUS_IGNORE);
  }

  MPI_Bcast(aPart, sizeRowsForEachProc * N, MPI_DOUBLE, 0, comm1DRows);

  MPI_Bcast(bPart, sizeColsForEachProc * N, MPI_DOUBLE, 0, comm1DColumns);

  for (int i = 0; i < sizeRowsForEachProc; ++i) {
    for (int j = 0; j < sizeColsForEachProc; ++j) {
      for (int k = 0; k < N; ++k) {
        cPart[i * sizeColsForEachProc + j] += aPart[i * N + k] * bPart[k * sizeColsForEachProc + j];
      }
    }
  }

  collectCMatrix(dims, coords, c, cPart, sizeRowStrip, sizeRowStripMod, sizeColumnStrip, sizeColumnStripMod,rank);


  if (rank == 0) {
    free(sendCountsB);
    free(displsB);
    free(sendCountsA);
    free(displsA);
    MPI_Type_free(&typeB);
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

void createsTypes(MPI_Datatype *typeB, MPI_Datatype *typeBMod, int sizeColumnStrip, int sizeColumnStripMod) {

  MPI_Type_vector(N, sizeColumnStrip, K, MPI_DOUBLE, typeB);
  MPI_Type_vector(N, sizeColumnStrip + sizeColumnStripMod, K, MPI_DOUBLE, typeBMod);

  MPI_Type_create_resized(*typeB, 0, sizeColumnStrip * sizeof(double), typeB);

  MPI_Type_commit(typeB);
  MPI_Type_commit(typeBMod);

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

void collectCMatrix(int *dims, int *coords, double *c, double *cPart, int sizeRowStrip, int sizeRowStripMod,
                    int sizeColumnStrip, int sizeColumnStripMod,int rank) {
  int yTmp, xTmp;
  int recvCountsY[dims[0] * dims[1]];
  int recvCountsSplitY[dims[0] * dims[1]];
  int recvCountsX[dims[0] * dims[1]];
  int recvCountsSplitX[dims[0] * dims[1]];
  for (int i = 0; i < dims[0]; i++) {
    for (int j = 0; j < dims[1]; j++) {
      xTmp = (i == dims[0] - 1) ? (sizeRowStrip + sizeRowStripMod) : sizeRowStrip;
      yTmp = (j == dims[1] - 1) ? (sizeColumnStrip + sizeColumnStripMod) : sizeColumnStrip;
      recvCountsY[i * dims[1] + j] = yTmp * N;
      recvCountsX[i * dims[1] + j] = xTmp * N;
      recvCountsSplitX[i * dims[1] + j] = xTmp;
      recvCountsSplitY[i * dims[1] + j] = yTmp;
    }
  }
  int countA = recvCountsY[coords[1] * dims[1] + coords[1]];
  int splitX = recvCountsSplitX[coords[0] * dims[1] + coords[1]];
  int splitY = recvCountsSplitY[coords[0] * dims[1] + coords[1]];
  sendRecvMatrix(c,cPart,dims,recvCountsY,splitX,splitY,countA,rank);
}

void sendRecvMatrix(double *c, double *cPart, int *dims, int *recvCountsY, int splitX,
    int splitY, int countA,int rank) {
  if (rank == 0) {
    int index = 0;
    for (int i = 0; i < M * K; i++) {

      int trankY = (i % K) / (K / dims[1]);
      //Zero rank gets Y coord for recv and collect (factorization+count the coord)
      int trankX = ((i - (i % K)) / M) / (M / dims[0]);
      //Zero rank gets X coord for recv and collect (factorization+count the coord)

      if (trankX >= dims[0]) trankX = dims[0] - 1;//if the number goes beyond the dims => sub 1
      if (trankY >= dims[1]) trankY = dims[1] - 1;//if the number goes beyond the dims => sub 1

      int trank = trankX * dims[1] + trankY;

      if (trank != 0) {

        MPI_Recv(&c[i],
                 recvCountsY[trankX * dims[1] + trankY] / N,
                 MPI_DOUBLE,
                 trankX * dims[1] + trankY,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        i += (recvCountsY[trankX * dims[1] + trankY] / N) - 1;

      } else {
        c[i] = cPart[index++];
      }
    }
  } else {
    for (int i = 0; i < splitX * splitY; i += countA / N) {
      MPI_Send(&cPart[i], splitY, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);//Send a line from subprocess
    }
  }

}
