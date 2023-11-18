// GABRIEL PIMENTEL DOLZAN
// GRR20209948
//
// TULIO DE PADUA DUTRA
// GRR20206155

#include "heap.h"

void swap(pair_t *a, pair_t *b)
{
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

void maxHeapify(pair_t heap[], int size, int i)
{
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < size && heap[left].key > heap[largest].key)
    {
        largest = left;
    }

    if (right < size && heap[right].key > heap[largest].key)
    {
        largest = right;
    }

    if (largest != i)
    {
        swap(&heap[i], &heap[largest]);
        maxHeapify(heap, size, largest);
    }
}

void heapifyUp(pair_t heap[], int *size, int pos)
{
    int parent = (pos - 1) / 2;

    while (pos > 0 && heap[parent].key < heap[pos].key)
    {
        swap(&heap[parent], &heap[pos]);
        pos = parent;
        parent = (pos - 1) / 2;
    }
}

void insert(pair_t heap[], int *size, pair_t element)
{
    if (*size == 0)
    {
        heap[0] = element;
        (*size)++;
    }
    else
    {
        heap[*size] = element;
        (*size)++;
        heapifyUp(heap, size, *size - 1);
    }
}

void decreaseMax(pair_t heap[], int size, pair_t element)
{
    if (size == 0)
        return;

    if (heap[0].key > element.key)
    {
        heap[0].key = element.key;
        heap[0].val = element.val;
        maxHeapify(heap, size, 0);
    }
}
