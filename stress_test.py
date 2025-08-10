import random
import sys

import numpy as np
import pandas as pd
import torch

from bitsandbytes import functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ITERATIONS = 1000
WARMUP_ITER = 10


DEVICE = ["cuda"]
DTYPE = [torch.float32, torch.float16, torch.bfloat16]
QUANT_TYPE = ["fp4", "nf4"]
BLOCKSIZE = [64, 128, 256, 512, 1024, 2048, 4096]
TENSOR_SHAPE = [1024, 2048, 4096, 8192, 16384]

# metadata logger, used if benchmarking with ncu only
logger = {"tensor_shape": [], "blocksize": [], "quant_type": [], "is_warmup": []}


def sync_and_time(func, *args, **kwargs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    ret = func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    return elapsed, ret


def test_4bit_quantize(A1, quant_type, blocksize, warmup):
    logger["tensor_shape"].append(A1.shape[0])
    logger["blocksize"].append(blocksize)
    logger["quant_type"].append(quant_type)
    logger["is_warmup"].append(warmup)
    return sync_and_time(F.quantize_4bit, A1, blocksize=blocksize, quant_type=quant_type)


def test_4bit_dequantize(input_shape, qa, SA, quant_type, blocksize, warmup):
    logger["tensor_shape"].append(input_shape)
    logger["blocksize"].append(blocksize)
    logger["quant_type"].append(quant_type)
    logger["is_warmup"].append(warmup)
    return sync_and_time(F.dequantize_4bit, qa, SA, blocksize=blocksize, quant_type=quant_type)


def test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False):
    t1, (qa, SA) = test_4bit_quantize(A1, quant_type, blocksize, warmup)
    t2 = test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup)[0]
    return t1 + t2


def stress_test_2(A1, quant_type, blocksize, iterations=ITERATIONS, warmup=WARMUP_ITER):
    for _ in range(warmup):
        test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=True)

    elapsed = 0
    for _ in range(iterations):
        elapsed += test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False)
    return elapsed


def stress_test_3(A1, quant_type, blocksize, iterations=ITERATIONS, warmup=WARMUP_ITER):
    qa, SA = None, None
    for _ in range(warmup):
        qa, SA = test_4bit_quantize(A1, quant_type, blocksize, warmup=True)[1]

    elapsed = 0
    for _ in range(iterations):
        elapsed += test_4bit_quantize(A1, quant_type, blocksize, warmup=False)[0]

    for _ in range(warmup):
        test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup=True)

    elapsed2 = 0
    for _ in range(iterations):
        elapsed2 += test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup=False)[0]
    return elapsed, elapsed2


def save_metadata():
    df = pd.DataFrame(logger)
    df.to_csv("stress_test_metadata.csv")


def one_time_test():
    # for debugging only
    tensor_shape = 1024
    device = "cuda"
    dtype = torch.float32
    quant_type = "fp4"
    blocksize = 1024
    A1 = torch.randn(tensor_shape, tensor_shape, device=device, dtype=dtype)
    test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False)

    quant_type = "nf4"
    test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False)

    save_metadata()
    return


def main():
    # return one_time_test()
    results = []
    for device in DEVICE:
        for dtype in DTYPE:
            for quant_type in QUANT_TYPE:
                for blocksize in BLOCKSIZE:
                    for tensor_shape in TENSOR_SHAPE:
                        A1 = torch.randn(tensor_shape, tensor_shape, device=device, dtype=dtype)
                        print(f"[{device}-{tensor_shape}-{dtype}-{quant_type}-{blocksize}]: ", flush=True, end="")
                        elapsed1 = stress_test_2(A1, quant_type, blocksize, ITERATIONS, WARMUP_ITER)
                        elapsed2, elapsed3 = stress_test_3(A1, quant_type, blocksize, ITERATIONS, WARMUP_ITER)

                        results.append(
                            {
                                "dtype": dtype,
                                "quant_type": quant_type,
                                "blocksize": blocksize,
                                "tensor_shape": tensor_shape,
                                "quantize_dequantize_latency": elapsed1,
                                "quantize_latency": elapsed2,
                                "dequantize_latency": elapsed3,
                            }
                        )
                        print(
                            {
                                "quant_dequant_latency": elapsed1,
                                "quant_latency": elapsed2,
                                "dequant_latency": elapsed3,
                            }
                        )
    df = pd.DataFrame(results)
    df.to_csv(sys.argv[1])
    save_metadata()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stress_test.py <output.csv>")
        sys.exit()
    torch.cuda.empty_cache()
    main()
