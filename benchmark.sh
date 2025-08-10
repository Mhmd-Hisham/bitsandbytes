#!/bin/bash
# This script is used to benchmark the performance from a branch in my fork against the baseline repo of BNB

# the branch to benchmark
benchmark_branch="cuda-branchless-binary-search"

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"
export CUDA_LAUNCH_BLOCKING=1

mkdir benchmark_results

# clone my fork
git clone https://github.com/Mhmd-Hisham/bitsandbytes.git
cd bitsandbytes
git checkout "${benchmark_branch}"

# build for cuda
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=90 .
cmake --build build_cuda --config Release

nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism
nvidia-smi -c EXCLUSIVE_PROCESS        # restrict to only one process can create a cuda context on the GPU at any given time

# I chose these value based on my rtx 2060 mobile
# run nvidia-smi -q -d SUPPORTED_CLOCKS to know the possible ranges for your gpu
# doesn't work on windows nor Vast.ai
nvidia-smi -lgc 2100,2100              # set min and max graphics freq in MHz
nvidia-smi -lmc 5001                   # set memory freq in MHz

python stress_test.py "../benchmark_results/improved_kernel_run1.csv"
python stress_test.py "../benchmark_results/improved_kernel_run2.csv"
python stress_test.py "../benchmark_results/improved_kernel_run3.csv"

# copy the stress test
mv stress_test.py ..

# chdir and rename the fork
cd ..
mv bitsandbytes bitsandbytes_fork

# clone original repo
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git

# move the stress test to the baseline repo
mv stress_test.py bitsandbytes/

cd bitsandbytes

# build for cuda
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=90 .
cmake --build build_cuda --config Release

# benchmark the baseline repo
python stress_test.py "../benchmark_results/baseline_kernel_run1.csv"
python stress_test.py "../benchmark_results/baseline_kernel_run2.csv"
python stress_test.py "../benchmark_results/baseline_kernel_run3.csv"
mv stress_test_metadata.csv "../benchmark_results/stress_test_metadata.csv"
mv stress_test.py ../benchmark_results/stress_test.py

cd ..

# zip the results to download with scp
zip -r benchmark_results.zip benchmark_results
