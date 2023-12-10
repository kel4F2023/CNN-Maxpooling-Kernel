CC = mpicc
CFLAGS = -mavx2 -mfma -Wno-implicit-function-declaration -O3 -std=c99 -g
HEADERS = pack.c pooling.c utils.c

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

compile: $(OBJS)
	$(CC) $(CFLAGS) driver.c $(HEADERS) -o driver.x -march=native
	$(CC) $(CFLAGS) experiment.c  $(HEADERS) -o experiment.x -march=native
	$(CC) $(CFLAGS) perfmance.c  $(HEADERS) -o perfmance.x -march=native


run:
	mpiexec -n 1 ./experiment.x

test:
	mpiexec -n 1 ./driver.x

perf:
	perf stat ./perfmance.x 256
	
flamegraph:
	perf record -g  ./perfmance.x 256
	perf script | ../FlameGraph/stackcollapse-perf.pl > out.perf-folded
	../FlameGraph/flamegraph.pl out.perf-folded > mygraph.svg

clean:
	rm -f *.x *~ *.o