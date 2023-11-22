CC = mpicc
CFLAGS = -Wall -g -O3 # gerar "warnings" detalhados e infos de depuração

# regra default (primeira regra)
all: knn_mpi

objs = knn_mpi.o heap.o chrono.o

# regras de ligacao
knn_mpi: $(objs)
	$(CC) $(CFLAGS) -o $@ $^

# remove arquivos temporários
clean:
	-rm -f $(objs) *~

# remove tudo o que não for o código-fonte
purge: clean
	-rm -f knn_mpi