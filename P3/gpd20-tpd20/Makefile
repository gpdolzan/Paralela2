CC = mpicc
CFLAGS = -Wall -g -O3 -lpthread # gerar "warnings" detalhados e infos de depuração

# regra default (primeira regra)
all: knn_mpi_pthread

objs = knn_mpi_pthread.o heap.o chrono.o

# regras de ligacao
knn_mpi_pthread: $(objs)
	$(CC) $(CFLAGS) -o $@ $^

# remove arquivos temporários
clean:
	-rm -f $(objs) *~

# remove tudo o que não for o código-fonte
purge: clean
	-rm -f knn_mpi_pthread
