CC = gcc
CFLAGS = -O0 -Wall -msse2 -mavx2 -fno-stack-protector -mavx -std=gnu99
GIT_HOOKS := .git/hooks/pre-commit
EXEC = main_naive main_sse main_sse_prefetch main_AVX main_AVX_prefetch main_AVX_prefetchV2

default: impl.o 
	$(CC) $(CFLAGS) impl.o main.c -Dnaive -o main_naive
	$(CC) $(CFLAGS) impl.o main.c -Dsse -o  main_sse
	$(CC) $(CFLAGS) impl.o main.c -Dsse_prefetch -o main_sse_prefetch
	$(CC) $(CFLAGS) impl.o main.c -DAVX -o  main_AVX
	$(CC) $(CFLAGS) impl.o main.c -DAVX_prefetch -o main_AVX_prefetch
	$(CC) $(CFLAGS) impl.o main.c -DAVX_prefetchV2 -o main_AVX_prefetchV2
.PHONY: clean default

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@ 

check: default
	 ./main_naive  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	 ./main_sse  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	 ./main_sse_prefetch  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	 ./main_AVX  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	 ./main_AVX_prefetch  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	 ./main_AVX_prefetchV2  && echo 1 | sudo tee /proc/sys/vm/drop_caches

cache-test: $(EXEC)
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_naive > naive.txt && echo 1 | sudo tee /proc/sys/vm/drop_caches
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_sse   > sse.txt && echo 1 | sudo tee /proc/sys/vm/drop_caches
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_sse_prefetch > sse_prefetch.txt  && echo 1 | sudo tee /proc/sys/vm/drop_caches
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_AVX > AVX.txt && echo 1 | sudo tee /proc/sys/vm/drop_caches
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_AVX_prefetch  > AVX_prefetch.txt&& echo 1 | sudo tee /proc/sys/vm/drop_caches
	perf stat --repeat 100 \
		-e cache-misses,cache-references,instructions,cycles \
		./main_AVX_prefetchV2  > AVX_prefetchV2.txt&& echo 1 | sudo tee /proc/sys/vm/drop_caches

all: $(GIT_HOOKS) main.c
	$(CC) $(CFLAGS) -o main main.c

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

clean:
	rm -f $(EXEC) *.o *.s *.txt 
