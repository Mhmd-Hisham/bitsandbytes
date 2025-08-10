import pandas as pd
import torch
import time
from bitsandbytes import functional as F

SEED = 42
torch.manual_seed(SEED)

ITERATIONS = 100
WARMUP_ITER = 5

DEVICE = ["cuda"]
DTYPE = [torch.float32]  # , torch.float16, torch.bfloat16]
QUANT_TYPE = ["fp4", "nf4"]
BLOCKSIZE = [64, 128, 256, 512, 1024, 2048, 4096]
TENSOR_SHAPE = [1024, 2048, 4096, 8192, 16384]

# metadata logger
logger = {"tensor_shape": [], "blocksize": [], "quant_type": [], "is_warmup": []}


def test_4bit_quantize(A1, quant_type, blocksize, warmup):
    logger["tensor_shape"].append(A1.shape[0])
    logger["blocksize"].append(blocksize)
    logger["quant_type"].append(quant_type)
    logger["is_warmup"].append(warmup)
    torch.cuda.synchronize()
    qa, SA = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
    torch.cuda.synchronize()
    return qa, SA


def test_4bit_dequantize(input_shape, qa, SA, quant_type, blocksize, warmup):
    logger["tensor_shape"].append(input_shape)
    logger["blocksize"].append(blocksize)
    logger["quant_type"].append(quant_type)
    logger["is_warmup"].append(warmup)
    torch.cuda.synchronize()
    F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
    torch.cuda.synchronize()


def test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False):
    torch.cuda.synchronize()
    qa, SA = test_4bit_quantize(A1, quant_type, blocksize, warmup)
    torch.cuda.synchronize()
    test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup)
    torch.cuda.synchronize()


def stress_test_2(A1, quant_type, blocksize, iterations=ITERATIONS, warmup=WARMUP_ITER):
    for _ in range(warmup):
        test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=True)

    t = time.perf_counter()
    for _ in range(iterations):
        test_4bit_quantize_dequantize(A1, quant_type, blocksize, warmup=False)
    elasped = time.perf_counter() - t
    return elapsed

def stress_test_3(A1, quant_type, blocksize, iterations=ITERATIONS, warmup=WARMUP_ITER):
    qa, SA = None, None
    for _ in range(warmup):
        qa, SA = test_4bit_quantize(A1, quant_type, blocksize, warmup=True)

    t = time.perf_counter()
    for _ in range(iterations):
        test_4bit_quantize(A1, quant_type, blocksize, warmup=False)
    elapsed = time.perf_counter() - t

    for _ in range(warmup):
        test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup=True)

    t = time.perf_counter()
    for _ in range(iterations):
        test_4bit_dequantize(A1.shape[0], qa, SA, quant_type, blocksize, warmup=False)
    elapsed2 = time.perf_counter() - t
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
                        print(f"[{device}-{tensor_shape}-{dtype}-{quant_type}-{blocksize}]: ", flush=True)
                        elapsed1 = stress_test_2(A1, quant_type, blocksize, ITERATIONS, WARMUP_ITER)
                        elapsed2, elapsed3 = stress_test_3(A1, quant_type, blocksize, ITERATIONS, WARMUP_ITER)

                        results.append({
                            "dtype": dtype,
                            "quant_type": quant_type,
                            "blocksize": blocksize,
                            "tensor_shape": tensor_shape,
                            "quantize_dequantize_latency": elapsed1,
                            "quantize_latency": elapsed2,
                            "dequantize_latency": elapsed3,
                        })
    df = pd.DataFrame(results)
    df.to_csv("improved_results.csv")
    save_metadata()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
