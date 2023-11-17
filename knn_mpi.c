#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include "heap.h"
#include "chrono.c"

// Estrutura para um ponto em D dimensões
typedef struct {
    float *coordinates;  // Coordenadas do ponto
} Point;

// Function to generate a set of random points in a flattened array
void geraConjuntoDeDados(float *C, int nc, int D) {
    for (int i = 0; i < nc * D; i++) {
        C[i] = (float)rand() / RAND_MAX;
    }
}

// Function to calculate the squared distance between two points in flattened arrays
float squaredDistance(const float *p1, const float *p2, int D) {
    float distance = 0.0;
    for (int i = 0; i < D; i++) {
        float diff = p1[i] - p2[i];
        distance += diff * diff;
    }
    return distance;
}

// Function to perform the k-NN algorithm
void knn(float *Q, int nq, float *P, int npp, int D, int k, int **resultIndices) {
    for (int i = 0; i < nq; i++) {
        pair_t *neighborsHeap = (pair_t *)malloc(k * sizeof(pair_t));
        int heapSize = 0;

        for (int j = 0; j < npp; j++) {
            float distance = squaredDistance(&Q[i * D], &P[j * D], D);
            pair_t neighbor = {distance, j};

            if (heapSize < k) {
                insert(neighborsHeap, &heapSize, neighbor);
            } else if (distance < neighborsHeap[0].key) {
                decreaseMax(neighborsHeap, k, neighbor);
            }
        }

        for (int j = 0; j < k; j++) {
            resultIndices[i][j] = neighborsHeap[j].val;
        }

        free(neighborsHeap);
    }
}

// Function to verify the k-NN results
void verificaKNN(float *Q, int nq, float *P, int n, int D, int k, int *R) {
    int correto = 1;

    for (int i = 0; i < nq; i++) {
        float *distances = (float *)malloc(k * sizeof(float));
        for (int j = 0; j < k; j++) {
            distances[j] = FLT_MAX;
        }

        for (int j = 0; j < n; j++) {
            float dist = squaredDistance(&Q[i * D], &P[j * D], D);

            for (int l = k - 1; l >= 0; l--) {
                if (dist < distances[l]) {
                    if (l < k - 1) {
                        distances[l + 1] = distances[l];
                    }
                    distances[l] = dist;
                } else {
                    break;
                }
            }
        }

        for (int j = 0; j < k; j++) {
            float resultDist = squaredDistance(&Q[i * D], &P[R[i * k + j] * D], D);
            int found = 0;
            for (int l = 0; l < k; l++) {
                if (resultDist == distances[l]) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                correto = 0;
                break;
            }
        }

        free(distances);

        if (!correto) {
            printf("ERRADO\n");
            return;
        }
    }
    printf("CORRETO\n");
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Usage: %s <nq> <npp> <d> <k>\n", argv[0]);
        return 1;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    chronometer_t chronometer;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int nq = atoi(argv[1]);  // Number of points in Q
    int npp = atoi(argv[2]); // Number of points in P
    int d = atoi(argv[3]);   // Number of dimensions
    int k = atoi(argv[4]);   // Number of nearest neighbors

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

    int local_nq = nq / world_size;
    float *local_Q_flattened = (float *)malloc(local_nq * d * sizeof(float));
    MPI_Scatter(Q_flattened, local_nq * d, MPI_FLOAT, local_Q_flattened, local_nq * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int *localResultIndices = (int *)malloc(local_nq * k * sizeof(int));
    int **resultIndices2D = (int **)malloc(local_nq * sizeof(int *));
    for (int i = 0; i < local_nq; i++)
    {
        resultIndices2D[i] = &localResultIndices[i * k];
    }

    if(world_rank == 0)
    {
        chrono_reset(&chronometer);
        chrono_start(&chronometer);
    }

    knn(local_Q_flattened, local_nq, P, npp, d, k, resultIndices2D);

    int *gatheredResultIndices = NULL;
    if (world_rank == 0)
    {
        gatheredResultIndices = (int *)malloc(nq * k * sizeof(int));
    }

    MPI_Gather(localResultIndices, local_nq * k, MPI_INT, gatheredResultIndices, local_nq * k, MPI_INT, 0, MPI_COMM_WORLD);

    // Barrier to guarantee knn processing is already over
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        chrono_stop(&chronometer);

        double total_time_in_seconds = (double) chrono_gettotal(&chronometer) / 1000000000.0;
        double MOPs = (double)nq / total_time_in_seconds;
        printf("Tempo: %lf segundos\n", total_time_in_seconds);
        printf("Throughput: %lf segundos\n", MOPs);

        verificaKNN(Q_flattened, nq, P, npp, d, k, gatheredResultIndices);
        free(gatheredResultIndices);
        free(Q_flattened);
        free(resultIndices);
    }

    free(local_Q_flattened);
    free(localResultIndices);
    free(resultIndices2D);
    free(P);

    MPI_Finalize();
    return 0;
}