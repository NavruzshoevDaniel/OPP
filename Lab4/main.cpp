#include<stdio.h>
#include<math.h>
#include <mpi.h>

/*Количество ячеек вдоль координат x, y, z*/
#define Nx 70
#define Ny 70
#define Nz 70
#define a 1e5

int I;
int J;
int K;

double phi(double, double, double);
double ro(double, double, double);
void initBounds(int *linesPerProc, int *offsets, int curProc);
void calcEdges();
void sendData();
void calcCenter();
void recData();
void findMaxDiff();

/* Выделение памяти для 3D пространства для текущей и предыдущей итерации */
double *(F[2]);
double *(buffer[2]);
double hx, hy, hz; //расстояния между соседними узлами сетки

double Fi, Fj, Fk, F1;
double Dx = 2.0;
double Dy = 2.0;
double Dz = 2.0;
double e = 1e-8;
int prev = 1;
int next = 0;
int f, tmpF;
int curProc = 0;
int nmbOfProc = 0;
double hXSqr;
double hYSqr;
double hZSqr;

int *linesPerProc;
int *offsets;

double c;
MPI_Request sendRequest[2] = {}; //идентификатор для асинхронного приема сообщений
MPI_Request recRequest[2] = {}; //идентификатор для асинхронной передачи сообщений


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nmbOfProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &curProc);

  int height = Nx;
  int tmp = nmbOfProc - (height % nmbOfProc); //номер процесса, начиная с которого дается +1 строка
  int currentLine = 0;

  linesPerProc = new int[nmbOfProc]();
  offsets = new int[nmbOfProc](); //количество строк в предыдущих процессах

  for (int i = 0; i < nmbOfProc; ++i) {
    offsets[i] = currentLine;
    if (i < tmp) {
      linesPerProc[i] = height / nmbOfProc;
    } else {
      linesPerProc[i] = height / nmbOfProc + 1;
    }
    currentLine += linesPerProc[i];
  }

  I = linesPerProc[curProc];
  J = Ny ;
  K = Nz;

  F[0] = new double[I * J * K]();
  F[1] = new double[I * J * K]();

  buffer[0] = new double[K * J]();
  buffer[1] = new double[K * J]();

  /* Размеры шагов */
  hx = Dx / (Nx-1);
  hy = Dy / (Ny-1);
  hz = Dz / (Nz-1);

  hXSqr = hx * hx;
  hYSqr = hy * hy;
  hZSqr = hz * hz;
  c = 2 / hXSqr + 2 / hYSqr + 2 / hZSqr + a;

  initBounds(linesPerProc, offsets, curProc);

  double start = MPI_Wtime();

  do {
    f = 1;
    prev = 1 - prev;
    next = 1 - next;

    //обмениваемся краями
    sendData();

    //считаем середину
    calcCenter();


    //Ждем получения всех данных
    recData();

    //считаем края
    calcEdges();

    MPI_Allreduce(&f, &tmpF, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    f = tmpF;
  } while (f == 0);

  double finish = MPI_Wtime();

  if (curProc == 0) {
    printf("Time: %lf\n", finish - start);
  }

  findMaxDiff();

  delete[] buffer[0];
  delete[] buffer[1];
  delete[] F[0];
  delete F[1];
  delete[] offsets;
  delete[] linesPerProc;

  MPI_Finalize();
  return 0;
}

/* Функция определения точного решения*/
double phi(double x, double y, double z) {
  return x * x + y * y + z * z;
}

/* Функция задания правой части уравнения */
double ro(double x, double y, double z) {
  return (6 - a * phi(x, y, z));
}

void initBounds(int *linesPerProc, int *offsets, int curProc) {
  for (int i = 0, startLine = offsets[curProc]; i <= linesPerProc[curProc] - 1; i++, startLine++) {
    for (int j = 0; j <= Ny-1; j++) {
      for (int k = 0; k <= Nz-1; k++) {
        if ((startLine != 0) && (j != 0) && (k != 0) && (startLine != Nx-1) && (j != Ny-1) && (k != Nz-1)) {
          F[0][i * J * K + j * K + k] = 0;
          F[1][i * J * K + j * K + k] = 0;
        } else {
          F[0][i * J * K + j * K + k] = phi(startLine * hx, j * hy, k * hz); //граничные условия
          F[1][i * J * K + j * K + k] = F[0][i * J * K + j * K + k];
        }
      }
    }
  }
}

void calcEdges() {
  for (int j = 1; j < Ny-1; ++j) {
    for (int k = 1; k < Nz-1; ++k) {
      if (curProc != 0) {
        int i = 0;
        Fi = (F[prev][(i + 1) * J * K + j * K + k] + buffer[0][j * K + k]) / hXSqr;
        Fj = (F[prev][i * J * K + (j + 1) * K + k] + F[prev][i * J * K + (j - 1) * K + k]) / hYSqr;
        Fk = (F[prev][i * J * K + j * K + (k + 1)] + F[prev][i * J * K + j * K + (k - 1)]) / hZSqr;
        F[next][i * J * K + j * K + k] = (Fi + Fj + Fk - ro((i + offsets[curProc]) * hx, j * hy, k * hz)) / c;
        if (fabs(F[next][i * J * K + j * K + k] - F[prev][i * J * K + j * K + k]) > e) {
          f = 0;
        }
      }
      if (curProc != nmbOfProc - 1) {
        int i = linesPerProc[curProc] - 1;
        Fi = (buffer[1][j * K + k] + F[prev][(i - 1) * J * K + j * K + k]) / hXSqr;
        Fj = (F[prev][i * J * K + (j + 1) * K + k] + F[prev][i * J * K + (j - 1) * K + k]) / hYSqr;
        Fk = (F[prev][i * J * K + j * K + (k + 1)] + F[prev][i * J * K + j * K + (k - 1)]) / hZSqr;
        F[next][i * J * K + j * K + k] = (Fi + Fj + Fk - ro((i + offsets[curProc]) * hx, j * hy, k * hz)) / c;
        if (fabs(F[next][i * J * K + j * K + k] - F[prev][i * J * K + j * K + k]) > e) {
          f = 0;
        }
      }
    }
  }
}

void sendData() { //на фоне
  if (curProc != 0) {//1
    /*передача сообщения без блокировки, обменялись верхними слоями*/
    MPI_Isend(&(F[prev][0]), K * J, MPI_DOUBLE, curProc - 1, 0, MPI_COMM_WORLD, &sendRequest[0]); //низ
    MPI_Irecv(buffer[0], K * J, MPI_DOUBLE, curProc - 1, 1, MPI_COMM_WORLD, &recRequest[1]);
  }
  if (curProc != nmbOfProc - 1) { //0
/*передача сообщения без блокировки, обмнялись нижними слоями*/
    MPI_Isend(&(F[prev][(linesPerProc[curProc] - 1) * J * K]),
              K * J,
              MPI_DOUBLE,
              curProc + 1,
              1,
              MPI_COMM_WORLD,
              &sendRequest[1]); //верх
    MPI_Irecv(buffer[1], K * J, MPI_DOUBLE, curProc + 1, 0, MPI_COMM_WORLD, &recRequest[0]);
  }
}

void calcCenter() {
  for (int i = 1; i < linesPerProc[curProc] - 1; ++i) {
    for (int j = 1; j < Ny-1; ++j) {
      for (int k = 1; k < Nz-1; ++k) {
        Fi = (F[prev][(i + 1) * J * K + j * K + k] + F[prev][(i - 1) * J * K + j * K + k]) / hXSqr;
        Fj = (F[prev][i * J * K + (j + 1) * K + k] + F[prev][i * J * K + (j - 1) * K + k]) / hYSqr;
        Fk = (F[prev][i * J * K + j * K + (k + 1)] + F[prev][i * J * K + j * K + (k - 1)]) / hZSqr;
        F[next][i * J * K + j * K + k] = (Fi + Fj + Fk - ro((i + offsets[curProc]) * hx, j * hy, k * hz)) / c;
        if (fabs(F[next][i * J * K + j * K + k] - F[prev][i * J * K + j * K + k]) > e) {
          f = 0;
        }
      }
    }
  }
}

void recData() {
  if (curProc != 0) {
    MPI_Wait(&recRequest[1], MPI_STATUS_IGNORE);
    MPI_Wait(&sendRequest[0], MPI_STATUS_IGNORE);
  }
  if (curProc != nmbOfProc - 1) {
    MPI_Wait(&recRequest[0], MPI_STATUS_IGNORE);
    MPI_Wait(&sendRequest[1], MPI_STATUS_IGNORE);
  }
}

void findMaxDiff() {
  double max = 0.0;
  for (int i = 0; i < linesPerProc[curProc]; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        if(i==linesPerProc[curProc]-2){
          printf("[%d][%d][%d] =%.18f    =%.18f\n",i,j,k,F[next][i * J * K + j * K + k],phi((i + offsets[curProc]) * hx, j * hy, k * hz));
        }
        if ((F1 = fabs(F[next][i * J * K + j * K + k] - phi((i + offsets[curProc]) * hx, j * hy, k * hz))) > max) {
          max = F1;
        }
      }
    }
  }
  printf("max:%.18lf\n", max);
  double tmpMax = 0;
  MPI_Allreduce(&max, &tmpMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (curProc == 0) {
    //std::cout<<"Max differ: "<<tmpMax<<std::endl;
    printf("Max differ:%.15lf\n", tmpMax);
  }
}
