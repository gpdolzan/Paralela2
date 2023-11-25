// GABRIEL PIMENTEL DOLZAN
// GRR20209948
//
// TULIO DE PADUA DUTRA
// GRR20206155

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <pthread.h>
#include "heap.h"
#include "chrono.h"

// Estrutura para um ponto em D dimensões
typedef struct
{
    float *coordinates; // Coordenadas do ponto
} Point;


// Definição da estrutura de argumentos para a thread
typedef struct
{
    float *Q;
    int start_q;
    int end_q;
    float *P;
    int npp;
    int D;
    int k;
    int **resultIndices;
} knn_thread_args;

// Function to generate a set of random points in a flattened array
void geraConjuntoDeDados(float *C, int nc, int D)
{
    for (int i = 0; i < nc * D; i++)
    {
        C[i] = (float)rand() / RAND_MAX;
    }
}

// Function to calculate the squared distance between two points in flattened arrays
float squaredDistance(const float *p1, const float *p2, int D)
{
    float distance = 0.0;
    for (int i = 0; i < D; i++)
    {
        float diff = p1[i] - p2[i];
        distance += diff * diff;
    }
    return distance;
}

// Função de trabalho para cada thread
void *knn_thread_func(void *args)
{
    knn_thread_args *data = (knn_thread_args *)args;

    // Itera sobre o subconjunto de Q atribuído a esta thread
    for (int i = data->start_q; i < data->end_q; i++) {
        pair_t *neighborsHeap = (pair_t *)malloc(data->k * sizeof(pair_t));
        int heapSize = 0;

        // Itera sobre todos os pontos em P
        for (int j = 0; j < data->npp; j++) {
            float distance = squaredDistance(&data->Q[i * data->D], &data->P[j * data->D], data->D);
            pair_t neighbor = {distance, j};

            // Constrói um heap dos k vizinhos mais próximos
            if (heapSize < data->k) {
                insert(neighborsHeap, &heapSize, neighbor);
            } else if (distance < neighborsHeap[0].key) {
                decreaseMax(neighborsHeap, data->k, neighbor);
            }
        }

        // Copia os índices dos k vizinhos mais próximos para resultIndices
        for (int j = 0; j < data->k; j++) {
            data->resultIndices[i][j] = neighborsHeap[j].val;
        }

        free(neighborsHeap);
    }

    return NULL;
}

// Function to perform the k-NN algorithm
void knn(float *Q, int nq, float *P, int npp, int D, int k, int **resultIndices, int nt)
{
    pthread_t threads[nt];
    knn_thread_args args[nt];

    int per_thread = nq / nt;
    int extra = nq % nt;

    for (int i = 0; i < nt; i++) {
        args[i].Q = Q;
        args[i].start_q = i * per_thread + (i < extra ? i : extra);
        args[i].end_q = args[i].start_q + per_thread + (i < extra ? 1 : 0);
        args[i].P = P;
        args[i].npp = npp;
        args[i].D = D;
        args[i].k = k;
        args[i].resultIndices = resultIndices;

        pthread_create(&threads[i], NULL, knn_thread_func, (void *)&args[i]);
    }

    for (int i = 0; i < nt; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Function to verify the k-NN results
void verificaKNN(float *Q, int nq, float *P, int n, int D, int k, int *R)
{
    int correto = 1;

    for (int i = 0; i < nq; i++)
    {
        float *distances = (float *)malloc(k * sizeof(float));
        for (int j = 0; j < k; j++)
        {
            distances[j] = FLT_MAX;
        }

        for (int j = 0; j < n; j++)
        {
            float dist = squaredDistance(&Q[i * D], &P[j * D], D);

            for (int l = k - 1; l >= 0; l--)
            {
                if (dist < distances[l])
                {
                    if (l < k - 1)
                    {
                        distances[l + 1] = distances[l];
                    }
                    distances[l] = dist;
                }
                else
                {
                    break;
                }
            }
        }

        for (int j = 0; j < k; j++)
        {
            float resultDist = squaredDistance(&Q[i * D], &P[R[i * k + j] * D], D);
            int found = 0;
            for (int l = 0; l < k; l++)
            {
                if (resultDist == distances[l])
                {
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                correto = 0;
                break;
            }
        }

        free(distances);

        if (!correto)
        {
            printf("ERRADO\n");
            return;
        }
    }
    printf("CORRETO\n");
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("Usage: %s <nq> <npp> <d> <k> <nt>\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    chronometer_t chronometer;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int nq = atoi(argv[1]);  // Number of points in Q
    int npp = atoi(argv[2]); // Number of points in P
    int d = atoi(argv[3]);   // Number of dimensions
    int k = atoi(argv[4]);   // Number of nearest neighbors
    int nt = atoi(argv[5]);  // Number of threads per MPI process

    float *P = (float *)malloc(npp * d * sizeof(float));
    geraConjuntoDeDados(P, npp, d);

    float *Q_flattened = NULL;
    int *resultIndices = NULL;
    if (world_rank == 0)
    {
        Q_flattened = (float *)malloc(nq * d * sizeof(float));
        geraConjuntoDeDados(Q_flattened, nq, d);

        resultIndices = (int *)malloc(nq * k * sizeof(int));
    }

    // Compute the number of elements each process will handle
    int *sendcounts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));
    int sum = 0;
    for (int i = 0; i < world_size; i++)
    {
        sendcounts[i] = (nq / world_size) * d;
        if (i < nq % world_size)
        {
            sendcounts[i] += d;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_nq = sendcounts[world_rank] / d;
    float *local_Q_flattened = (float *)malloc(local_nq * d * sizeof(float));

    MPI_Scatterv(Q_flattened, sendcounts, displs, MPI_FLOAT, local_Q_flattened, sendcounts[world_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    int *localResultIndices = (int *)malloc(local_nq * k * sizeof(int));
    int **resultIndices2D = (int **)malloc(local_nq * sizeof(int *));
    for (int i = 0; i < local_nq; i++)
    {
        resultIndices2D[i] = &localResultIndices[i * k];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        chrono_reset(&chronometer);
        chrono_start(&chronometer);
    }

    knn(local_Q_flattened, local_nq, P, npp, d, k, resultIndices2D, nt);

    if (world_rank == 0)
    {
        chrono_stop(&chronometer);
    }

    int *recvcounts = malloc(world_size * sizeof(int));
    int *rdispls = malloc(world_size * sizeof(int));
    sum = 0;
    for (int i = 0; i < world_size; i++)
    {
        recvcounts[i] = sendcounts[i] / d * k;
        rdispls[i] = sum;
        sum += recvcounts[i];
    }

    int *gatheredResultIndices = NULL;
    if (world_rank == 0)
    {
        gatheredResultIndices = (int *)malloc(nq * k * sizeof(int));
    }

    MPI_Gatherv(localResultIndices, recvcounts[world_rank], MPI_INT, gatheredResultIndices, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        double total_time_in_seconds = (double)chrono_gettotal(&chronometer) / 1000000000.0;
        double MOPs = (double)npp / (total_time_in_seconds * 1000);
        printf("Tempo: %lf segundos\n", total_time_in_seconds);
        printf("Throughput: %lf MOPs/segundo\n", MOPs);

        verificaKNN(Q_flattened, nq, P, npp, d, k, gatheredResultIndices);
        free(gatheredResultIndices);
        free(Q_flattened);
        free(resultIndices);
    }

    free(local_Q_flattened);
    free(localResultIndices);
    free(resultIndices2D);
    free(P);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(rdispls);

    MPI_Finalize();
    return 0;
}