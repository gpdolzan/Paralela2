// GABRIEL PIMENTEL DOLZAN
// GRR20209948
//
// TULIO DE PADUA DUTRA
// GRR20206155

#ifndef HEAP_H
#define HEAP_H

#include <stdio.h>
#include <stdlib.h>

typedef struct pair_t
{
    float key;
    int val;
} pair_t;

void swap(pair_t *a, pair_t *b);
void maxHeapify(pair_t heap[], int size, int i);
void heapifyUp(pair_t heap[], int *size, int pos);
void insert(pair_t heap[], int *size, pair_t element);
void decreaseMax(pair_t heap[], int size, pair_t element);

#endif // HEAP_H