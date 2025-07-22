#include <BinSearch.h>

#ifndef common
#define common

using namespace BinSearch;

struct quantize_block_args {
    BinAlgo<Scalar, float, Direct2>* bin_searcher;
    float* code;
    float* A;
    float* absmax;
    unsigned char* out;
    long long block_end;
    long long block_idx;
    long long threadidx;
    long long blocksize;
};

struct dequantize_block_args {
    float* code;
    unsigned char* A;
    float* absmax;
    float* out;
    long long block_end;
    long long block_idx;
    long long threadidx;
    long long blocksize;
};

void quantize_block(const quantize_block_args& args);
void dequantize_block(const dequantize_block_args& args);

#endif
