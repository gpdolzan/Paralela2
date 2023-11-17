#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include "heap.h"

// Estrutura para um ponto em D dimensões
typedef struct {
    float *coordinates;  // Coordenadas do ponto
} Point;

// Função para gerar um conjunto de pontos aleatórios
void geraConjuntoDeDados(Point *C, int nc, int D)
{
    // Alocação de toda a memória necessária de uma vez
    float *allCoordinates = (float *)malloc(nc * D * sizeof(float));

    // Inicialização de pontos com blocos de memória
    for (int i = 0; i < nc; i++)
    {
        C[i].coordinates = &allCoordinates[i * D];
    }

    // Preenchimento dos pontos com números aleatórios
    for (int i = 0; i < nc * D; i++)
    {
        allCoordinates[i] = (float)rand() / RAND_MAX;
    }
}

// Função para calcular a distância quadrática entre dois pontos
float squaredDistance(const Point *p1, const Point *p2, int D)
{
    float distance = 0.0;
    for (int i = 0; i < D; i++)
    {
        float diff = p1->coordinates[i] - p2->coordinates[i];
        distance += diff * diff;
    }
    return distance;
}

// Função principal k-NN
void knn(Point *Q, int nq, Point *P, int n, int D, int k, int **resultIndices) {
    for (int i = 0; i < nq; i++)
    {
        // Heap para armazenar os k vizinhos mais próximos de Q[i]
        pair_t *neighborsHeap = (pair_t *)malloc(k * sizeof(pair_t));
        int heapSize = 0;

        for (int j = 0; j < n; j++)
        {
            float distance = squaredDistance(&Q[i], &P[j], D);
            pair_t neighbor;
            neighbor.key = distance; // A chave é a distância
            neighbor.val = j;        // O valor é o índice do ponto em P

            // Se o heap ainda não está cheio, insira o novo vizinho
            if (heapSize < k)
            {
                insert(neighborsHeap, &heapSize, neighbor);
            }
            // Se o heap está cheio, mas o novo vizinho está mais perto do que o mais distante no heap
            else if (distance < neighborsHeap[0].key)
            {
                decreaseMax(neighborsHeap, k, neighbor);
            }
        }

        // Armazena os índices dos k vizinhos mais próximos para Q[i]
        for (int j = 0; j < k; j++)
        {
            resultIndices[i][j] = neighborsHeap[j].val;
        }

        free(neighborsHeap);
    }
}

void verificaSolucao(Point *Q, int nq, Point *P, int n, int D, int k, int **resultIndices)
{
    int correto = 1;

    for (int i = 0; i < nq; i++)
    {
        // Array para armazenar as distâncias dos k vizinhos mais próximos
        float *distances = (float *)malloc(k * sizeof(float));
        for (int j = 0; j < k; j++)
        {
            distances[j] = FLT_MAX;
        }

        // Encontrar os k vizinhos mais próximos para Q[i]
        for (int j = 0; j < n; j++)
        {
            float dist = squaredDistance(&Q[i], &P[j], D);

            // Verifica se esta distância é menor que a maior distância no array distances
            if (dist < distances[k - 1])
            {
                distances[k - 1] = dist;

                // Ordena o array distances
                for (int l = k - 1; l > 0 && distances[l] < distances[l - 1]; l--)
                {
                    float temp = distances[l];
                    distances[l] = distances[l - 1];
                    distances[l - 1] = temp;
                }
            }
        }

        // Verifica se os resultados obtidos estão corretos
        for (int j = 0; j < k; j++)
        {
            float resultDist = squaredDistance(&Q[i], &P[resultIndices[i][j]], D);
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
            break;
        }
    }

    if (correto)
    {
        printf("CORRETO\n");
    } else
    {
        printf("ERRADO\n");
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Uso: %s <nq> <npp> <d> <k>\n", argv[0]);
        return 1;
    }

    Point *Q, *P;
    int **resultIndices;
    int nq, npp, d, k;

    nq = atoi(argv[1]);  // Número de pontos em Q
    npp = atoi(argv[2]); // Número de pontos em P
    d = atoi(argv[3]);   // Número de dimensões
    k = atoi(argv[4]);   // Número de vizinhos mais próximos

    // Alocação e geração de Q e P
    Q = (Point *)malloc(nq * sizeof(Point));
    P = (Point *)malloc(npp * sizeof(Point));
    geraConjuntoDeDados(Q, nq, d);
    geraConjuntoDeDados(P, npp, d);

    // Alocação de resultIndices
    resultIndices = (int **)malloc(nq * sizeof(int*));
    for (int i = 0; i < nq; i++) {
        resultIndices[i] = (int *)malloc(k * sizeof(int));
    }

    // Chamada da função k-NN
    knn(Q, nq, P, npp, d, k, resultIndices);

    // Por exemplo, você pode imprimir os resultados ou verificar a solução
    // verificaSolucao(Q, nq, P, npp, d, k, resultIndices);

    // Liberação de memória
    for (int i = 0; i < nq; i++) {
        free(resultIndices[i]);
    }
    free(resultIndices);
    free(Q);
    free(P);

    return 0;
}