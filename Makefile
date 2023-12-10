CC = mpicc
CFLAGS = -mavx2 -mfma -Wno-implicit-function-declaration -O3 -std=c99 -g
HEADERS = pack.c pooling.c utils.c

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

compile: $(OBJS)
	$(CC) $(CFLAGS) driver.c $(HEADERS) -o driver.x -march=native
	$(CC) $(CFLAGS) experiment.c  $(HEADERS) -o experiment.x -march=native
	$(CC) $(CFLAGS) experiment.c  $(HEADERS) -o mid-experiment.x -march=native -D OLD
	$(CC) $(CFLAGS) perfmance.c  $(HEADERS) -o perfmance.x -march=native


run:
	mpiexec -n 1 ./experiment.x

test:
	mpiexec -n 1 ./driver.x

perf:
	perf stat ./perfmance.x 224


data:
	./experiment.x | sed '1d' >> data/final-data.tsv
	./mid-experiment.x | sed '1d' >> data/mid-data.tsv

clean:
	rm -f *.x *~ *.o