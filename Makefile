CC = gcc
CFLAGS = -mavx2 -mfma -Wno-implicit-function-declaration -O3 -std=c99

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

compile: $(OBJS)
	$(CC) $(CFLAGS) driver.c  -o driver.x -march=native
	$(CC) $(CFLAGS) experiment.c  -o experiment.x -march=native


run:
	./experiment.x

test:
	./driver.x

clean:
	rm -f *.x *~ *.o