#include <stdio.h>

void naive_transpose(int *src, int *dst, int w, int h);
void sse_transpose(int *src, int *dst, int w, int h);
void AVX_transpose(int *src, int *dst, int w, int h);
void sse_prefetch_transpose(int *src, int *dst, int w, int h);
void AVX_prefetch_transpose(int *src, int *dst, int w, int h);
//void AVX_prefetch_transposeV2(int *src, int *dst, int w, int h);
