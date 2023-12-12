CC = mpicc
CFLAGS = -mavx2 -mfma -Wno-implicit-function-declaration -O3 -std=c99 -g
HEADERS = pack.c pooling.c utils.c

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

compile: $(OBJS)
	$(CC) $(CFLAGS) driver.c $(HEADERS) -o driver.x -march=native
	$(CC) $(CFLAGS) experiment.c  $(HEADERS) -o experiment.x -march=native
	$(CC) $(CFLAGS) experiment.c  $(HEADERS) -o mid-experiment.x -march=native -D OLD
	$(CC) $(CFLAGS) performance.c  $(HEADERS) -o performance.x -march=native
	$(CC) $(CFLAGS) performance.c  $(HEADERS) -o mid-performance.x -march=native -D OLD


run:
	mpiexec -n 1 ./experiment.x

test:
	mpiexec -n 1 ./driver.x

perf:
	perf stat ./performance.x 192
	perf stat ./mid-performance.x 192



data:
	./experiment.x | sed '1d' >> data/final-data.tsv
	./mid-experiment.x | sed '1d' >> data/mid-data.tsv

init-data:
	rm -f data/final-data.tsv
	rm -f data/mid-data.tsv
	./experiment.x >> data/final-data.tsv
	./mid-experiment.x >> data/mid-data.tsv

clean:
	rm -f *.x *~ *.o