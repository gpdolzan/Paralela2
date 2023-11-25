#!/bin/bash
mpirun -N 4 ./knn_mpi_pthread 128 400000 300 1024 2
