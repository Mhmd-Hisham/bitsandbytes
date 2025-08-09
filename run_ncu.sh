#!/bin/bash

# set the test type (baseline/improved)
test_type="improved"

test_name="${test_type}_bnb_kernel_benchmark"

# set the testing device to be cuda only
export BNB_TEST_DEVICE="cuda"
export CUDA_LAUNCH_BLOCKING=1

# build for cuda
rm -rf build_cuda
cmake -B build_cuda -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=75 .
cmake --build build_cuda --config Release

nvidia-smi -pm 1                       # enable persistence mode, stop gpu from powering down when idle
nvidia-smi --auto-boost-default=0      # disable auto boost aka automatic frequency scaling mechanism
nvidia-smi -c EXCLUSIVE_PROCESS        # restrict to only one process can create a cuda context on the GPU at any given time

# I chose these value based on my rtx 2060 mobile
# run nvidia-smi -q -d SUPPORTED_CLOCKS to know the possible ranges for your gpu
# doesn't work on windows
nvidia-smi -lgc 2100,2100              # set min and max graphics freq in MHz
nvidia-smi -lmc 5001                   # set memory freq in MHz

# define the metrics to include in the ncu report
metrics="gpu__time_duration_measured_user,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_branch_instructions_executed.sum,smsp__sass_branch_targets_threads_divergent.sum,sm__achieved_occupancy.avg.pct_of_peak_sustained_active,inst_executed.sum,sm__warps_eligible_per_cycle.avg"
# Metrics explained:
#     gpu__time_duration_measured_user,`                         #  context-aware time duration in nanoseconds
#     sm__throughput.avg.pct_of_peak_sustained_elapsed,`         # avg streaming multiprocessor throughput as a percentage of peak sustained
#     dram__throughput.avg.pct_of_peak_sustained_elapsed,`       # avg memory throughput as a percentage of peak sustained
#     smsp__sass_branch_instructions_executed.sum,`              # total number of branch instructions executed.
#     smsp__sass_branch_targets_threads_divergent.sum,`          # number of threads that took divergent branches
#     sm__achieved_occupancy.avg.pct_of_peak_sustained_active,`  # actual occupancy achieved
#     inst_executed.sum,`                                        # total instructions executed
#     sm__warps_eligible_per_cycle.avg, `                        # warps eligible per cycle

kernels_regex='"regex:k(Quantize|Dequantize)Blockwise"'
report_name="${test_name}.ncu-rep"
csv_name="${test_name}.csv"

# profile the stress test with ncu
ncu -f \
    --target-processes all \
    --kernel-name "$kernels_regex" \
    --metrics "$metrics" \
    --export "$report_name" \
    python stress_test.py

# export the csv
ncu --import "$report_name" --csv --page raw > "$csv_name"