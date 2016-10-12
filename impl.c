#include <stdio.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include "impl.h"
#define PFDIST 8
#define PFDIST2 16
void naive_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            *(dst + x * h + y) = *(src + y * w + x);
}

void sse_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
            __m128i I0 = _mm_loadu_si128((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}

void AVX_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 8) {
        for (int y = 0; y < h; y += 8) {
            __m256i I0 = _mm256_loadu_si256((__m256i *)(src + (y + 0) * w + x));
            __m256i I1 = _mm256_loadu_si256((__m256i *)(src + (y + 1) * w + x));
            __m256i I2 = _mm256_loadu_si256((__m256i *)(src + (y + 2) * w + x));
            __m256i I3 = _mm256_loadu_si256((__m256i *)(src + (y + 3) * w + x));
            __m256i I4 = _mm256_loadu_si256((__m256i *)(src + (y + 4) * w + x));
            __m256i I5 = _mm256_loadu_si256((__m256i *)(src + (y + 5) * w + x));
            __m256i I6 = _mm256_loadu_si256((__m256i *)(src + (y + 6) * w + x));
            __m256i I7 = _mm256_loadu_si256((__m256i *)(src + (y + 7) * w + x));

            __m256i T0 = _mm256_unpacklo_epi32(I0, I1);
            __m256i T1 = _mm256_unpackhi_epi32(I0, I1);
            __m256i T2 = _mm256_unpacklo_epi32(I2, I3);
            __m256i T3 = _mm256_unpackhi_epi32(I2, I3);
            __m256i T4 = _mm256_unpacklo_epi32(I4, I5);
            __m256i T5 = _mm256_unpackhi_epi32(I4, I5);
            __m256i T6 = _mm256_unpacklo_epi32(I6, I7);
            __m256i T7 = _mm256_unpackhi_epi32(I6, I7);

            I0 = _mm256_unpacklo_epi64(T0, T2);
            I1 = _mm256_unpackhi_epi64(T0, T2);
            I2 = _mm256_unpacklo_epi64(T1, T3);
            I3 = _mm256_unpackhi_epi64(T1, T3);
            I4 = _mm256_unpacklo_epi64(T4, T6);
            I5 = _mm256_unpackhi_epi64(T4, T6);
            I6 = _mm256_unpacklo_epi64(T5, T7);
            I7 = _mm256_unpackhi_epi64(T5, T7);

            _mm256_storeu_si256((__m256i *)(dst + ((x + 0) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 1) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 2) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 3) * h) + y), I3);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 4) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 5) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 6) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 7) * h) + y), I3);

        }
    }
}




void sse_prefetch_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {

            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);

            __m128i I0 = _mm_loadu_si128 ((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128 ((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128 ((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128 ((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}


void AVX_prefetch_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 8) {
        for (int y = 0; y < h; y += 8) {

            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 4) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 5) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 6) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 7) *w + x, _MM_HINT_T1);

            __m256i I0 = _mm256_loadu_si256((__m256i *)(src + (y + 0) * w + x));
            __m256i I1 = _mm256_loadu_si256((__m256i *)(src + (y + 1) * w + x));
            __m256i I2 = _mm256_loadu_si256((__m256i *)(src + (y + 2) * w + x));
            __m256i I3 = _mm256_loadu_si256((__m256i *)(src + (y + 3) * w + x));
            __m256i I4 = _mm256_loadu_si256((__m256i *)(src + (y + 4) * w + x));
            __m256i I5 = _mm256_loadu_si256((__m256i *)(src + (y + 5) * w + x));
            __m256i I6 = _mm256_loadu_si256((__m256i *)(src + (y + 6) * w + x));
            __m256i I7 = _mm256_loadu_si256((__m256i *)(src + (y + 7) * w + x));

            __m256i T0 = _mm256_unpacklo_epi32(I0, I1);
            __m256i T1 = _mm256_unpackhi_epi32(I0, I1);
            __m256i T2 = _mm256_unpacklo_epi32(I2, I3);
            __m256i T3 = _mm256_unpackhi_epi32(I2, I3);
            __m256i T4 = _mm256_unpacklo_epi32(I4, I5);
            __m256i T5 = _mm256_unpackhi_epi32(I4, I5);
            __m256i T6 = _mm256_unpacklo_epi32(I6, I7);
            __m256i T7 = _mm256_unpackhi_epi32(I6, I7);

            I0 = _mm256_unpacklo_epi64(T0, T2);
            I1 = _mm256_unpackhi_epi64(T0, T2);
            I2 = _mm256_unpacklo_epi64(T1, T3);
            I3 = _mm256_unpackhi_epi64(T1, T3);
            I4 = _mm256_unpacklo_epi64(T4, T6);
            I5 = _mm256_unpackhi_epi64(T4, T6);
            I6 = _mm256_unpacklo_epi64(T5, T7);
            I7 = _mm256_unpackhi_epi64(T5, T7);

            _mm256_storeu_si256((__m256i *)(dst + ((x + 0) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 1) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 2) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 3) * h) + y), I3);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 4) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 5) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 6) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 7) * h) + y), I3);
        }
    }
}

void AVX_prefetchV2_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 8) {
        for (int y = 0; y < h; y += 8) {
            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);
            __m256i I0 = _mm256_loadu_si256((__m256i *)(src + (y + 0) * w + x));
            __m256i I1 = _mm256_loadu_si256((__m256i *)(src + (y + 1) * w + x));
            __m256i I2 = _mm256_loadu_si256((__m256i *)(src + (y + 2) * w + x));
            __m256i I3 = _mm256_loadu_si256((__m256i *)(src + (y + 3) * w + x));
            __m256i I4 = _mm256_loadu_si256((__m256i *)(src + (y + 4) * w + x));
            __m256i I5 = _mm256_loadu_si256((__m256i *)(src + (y + 5) * w + x));
            __m256i I6 = _mm256_loadu_si256((__m256i *)(src + (y + 6) * w + x));
            __m256i I7 = _mm256_loadu_si256((__m256i *)(src + (y + 7) * w + x));
            __m256i T0 = _mm256_unpacklo_epi32(I0, I1);
            __m256i T1 = _mm256_unpackhi_epi32(I0, I1);
            __m256i T2 = _mm256_unpacklo_epi32(I2, I3);
            __m256i T3 = _mm256_unpackhi_epi32(I2, I3);
            __m256i T4 = _mm256_unpacklo_epi32(I4, I5);
            __m256i T5 = _mm256_unpackhi_epi32(I4, I5);
            __m256i T6 = _mm256_unpacklo_epi32(I6, I7);
            __m256i T7 = _mm256_unpackhi_epi32(I6, I7);
            _mm_prefetch(src+(y + PFDIST + 4) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 5) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 6) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 7) *w + x, _MM_HINT_T1);
            I0 = _mm256_unpacklo_epi64(T0, T2);
            I1 = _mm256_unpackhi_epi64(T0, T2);
            I2 = _mm256_unpacklo_epi64(T1, T3);
            I3 = _mm256_unpackhi_epi64(T1, T3);
            I4 = _mm256_unpacklo_epi64(T4, T6);
            I5 = _mm256_unpackhi_epi64(T4, T6);
            I6 = _mm256_unpacklo_epi64(T5, T7);
            I7 = _mm256_unpackhi_epi64(T5, T7);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 0) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 1) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 2) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 3) * h) + y), I3);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 4) * h) + y), I0);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 5) * h) + y), I1);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 6) * h) + y), I2);
            _mm256_storeu_si256((__m256i *)(dst + ((x + 7) * h) + y), I3);
            _mm_prefetch(src+(y + PFDIST2 + 0) *w + x, _MM_HINT_T2);
        }
    }
}


