#include<stdio.h>
#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <cmath>

#define NUM_TASKS 99
#define NUM_LISTS 3
#define WAITING 15000
int const STOP_RECEIVING = -1;

int globalTasksDone = 0;

pthread_mutex_t mutex;
int curTask = 0;
int leftTasks = 0;
int *tasks;
int localTasks = 0;

void *recvTasks(void *me);
void exec(int *tasks, int numTasks, int rank);

void shift(int rank) {
  for (int i = 0; i < rank; ++i) {
    printf("\t\t");
  }
}

void initTasks(int *tasks, int num, int rank, int size, int iterCounter);
int main(int argc, char **argv) {
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pthread_attr_t attrs;
  pthread_t recv;

  pthread_attr_init(&attrs);
  pthread_mutex_init(&mutex, NULL);
  pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

  int *sendData = (int *) calloc(1, sizeof(int));
  sendData[1] = rank;
  pthread_create(&recv, &attrs, recvTasks, sendData);
  pthread_attr_destroy(&attrs);

  int numTasksForEachProc = 0;

  if (rank < NUM_TASKS % size) {
    numTasksForEachProc = NUM_TASKS / size + 1;
  } else {
    numTasksForEachProc = NUM_TASKS / size;
  }
  leftTasks = numTasksForEachProc;

  printf("Proc #%d has started\n", rank);
  printf("Proc #%d has %d tasks\n ", rank, numTasksForEachProc);
  tasks = (int *) calloc(numTasksForEachProc, sizeof(int));

  int numList = 0;
  int extraExec = 1;
  int numExtraExec = 0;
  MPI_Status status;
  double start=MPI_Wtime();
  double startIter=0;
  double iterTime=0;
  double minIterTime=0;
  double maxIterTime=0;
  while (numList != NUM_LISTS) {

    initTasks(tasks, numTasksForEachProc, rank, size, numList);
    startIter=MPI_Wtime();
    shift(rank);
    printf("PROC #%d HAS STARTED EXECUTION\n ", rank);
    exec(tasks, numTasksForEachProc, rank);
    shift(rank);
    int k = 0;
    printf("PROC #%d HAS JUST FINISHED (local tasks done=%d)(TIME=%f)\n", rank, localTasks,MPI_Wtime()-startIter);
    while (extraExec) {
      extraExec = 0;

      for (int i = (rank + 1) % size; i != rank; i = (i + 1) % size) {
        shift(rank);
        printf("PROC #%d SEND TO #%d k=%d\n", rank, i, k++);
        MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Recv(&numExtraExec, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
        shift(rank);
        printf("PROC #%d WANT TO EXEC EXTRA TASKS(number of extra tasks=%d\n)", rank, numExtraExec);
        if (numExtraExec != 0) {
          MPI_Recv(tasks, numExtraExec, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
          shift(rank);
          printf("PROC #%d GET %d FROM #%d\n", rank, numExtraExec, i);
          for (int j = 0; j < numExtraExec; ++j) {
            shift(rank);
            printf("task[%d]=%d\n", j, tasks[j]);
          }
          shift(rank);
          printf("PROC #%d STARTED WORKING AGAIN\n", rank);
          exec(tasks, numExtraExec, rank);
          extraExec = 1;
          shift(rank);
          printf("PROC #%d HAS JUST FINISHED AGAIN(local tasks done=%d\n)", rank, localTasks);
        }
      }

    }
    iterTime=MPI_Wtime()-startIter;
    shift(rank);
    printf("END PROC #%d ITERATION (TIME=%f)\n", rank,iterTime);

    MPI_Allreduce(&iterTime,&minIterTime,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&iterTime,&maxIterTime,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    int global = 0;

    if(rank==0){
      printf("!!!DISBALANCE:%f!!!\n",maxIterTime-minIterTime);
      printf("!!!!PROPORTION OF DISBALANCE:%.3f%!!!!\n",(maxIterTime-minIterTime)/maxIterTime*100);
    }

    MPI_Allreduce(&globalTasksDone, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    shift(rank);
    printf("GLOBAL TASKS=%d\n", global);
    extraExec=1;
    numList++;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Send(&STOP_RECEIVING, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);

  printf("TIME:%f\n",MPI_Wtime()-start);
  pthread_join(recv, NULL);
  pthread_mutex_destroy(&mutex);

  MPI_Finalize();
  free(sendData);
  free(tasks);
  return 0;
}

void initTasks(int *tasks, int num, int rank, int size, int iterCounter) {
  for (int i = 0; i < num; ++i) {
    tasks[i] = abs(100-i%100)*abs(rank - (iterCounter % size)) * WAITING;
    printf("task=%d rank=%d ", tasks[i], rank);
  }
  printf("\n");
}

void *recvTasks(void *me) {
  int rank = *((int *) me);

  int receiving = 1;
  int sendTasks = 0;
  int recvRank = 0;

  MPI_Status status;
  while (receiving) {
    shift(rank);
    printf("PROC #%d WAITING RECV\n", rank);
    MPI_Recv(&recvRank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

    shift(rank);
    printf("PROC %d recvRank=%d\n", rank, recvRank);

    if (recvRank == STOP_RECEIVING) {
      pthread_exit(0);
    }

    pthread_mutex_lock(&mutex);
    if (leftTasks > 5) {
      shift(rank);
      printf("BEFORE rank=%d lefttask=%d curTask=%d\n", rank, leftTasks, curTask);

      sendTasks = leftTasks / 2;
      leftTasks -= sendTasks;
      shift(rank);
      printf("AFTER rank=%d lefttask=%d curTask=%d\n", rank, leftTasks, curTask);

      MPI_Send(&sendTasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);

      MPI_Send(&tasks[curTask+1], sendTasks, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      curTask += sendTasks;
      shift(rank);
      printf("SEND %d tasks from #%d to #%d\n", sendTasks, rank, status.MPI_SOURCE);
    } else {pthread_mutex_unlock(&mutex);
      sendTasks = 0;

      MPI_Send(&sendTasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      shift(rank);
      printf("AFTER rank=%d lefttask=%d\n", rank, leftTasks);

      shift(rank);
      printf("Nothing to send (proc #%d)\n", rank);
    }
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

void exec(int *tasks, int numTasks, int rank) {

  pthread_mutex_lock(&mutex);
  leftTasks = numTasks;
  curTask = 0;
  int localCurTask=0;

  while (leftTasks != 0) {
    leftTasks--;
    localTasks++;
    localCurTask=curTask;
    pthread_mutex_unlock(&mutex);

    globalTasksDone++;
    for (int j = 0; j < tasks[localCurTask]; ++j) {
      globalTasksDone *= (int) (sqrt((int) sqrt((int) sqrt(256))) / 2);
    }
    shift(rank);
    printf("TASK #%d PROC #%d DONE!(GLOBAL=%d)\n", localCurTask, rank, globalTasksDone);
    pthread_mutex_lock(&mutex);
    curTask++;
  }

  pthread_mutex_unlock(&mutex);
}
