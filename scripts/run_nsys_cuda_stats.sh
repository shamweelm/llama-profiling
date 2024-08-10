#!/bin/bash

# Variables
OUTPUT_DIR=$1
INPUT_PATH=$2
TASK_NAME=$3

# Ensure output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Output directory does not exist"
    mkdir -p $OUTPUT_DIR
fi

# Ensure input file exists
if [ ! -f "$INPUT_PATH" ]; then
  echo "Input file does not exist"
  exit 1
fi

# Get the nsys stats
# Commands
# Not needed
# cmd="nsys stats -r cuda_api_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "nvtx_gpu_proj_sum completed."

cmd="nsys stats -r cuda_api_trace -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
echo $cmd
$cmd
echo "cuda_api_trace completed."

# Not needed
# cmd="nsys stats -r cuda_gpu_kern_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_gpu_kern_sum completed."

# Not needed
# cmd="nsys stats -r cuda_gpu_kern_gb_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_gpu_kern_gb_sum completed."

# Not needed
# cmd="nsys stats -r cuda_gpu_mem_size_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_gpu_mem_size_sum completed."

# Not needed
# cmd="nsys stats -r cuda_gpu_mem_time_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_gpu_mem_time_sum completed."

# Not needed
# cmd="nsys stats -r cuda_gpu_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_gpu_sum completed."

cmd="nsys stats -r cuda_gpu_trace -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
echo $cmd
$cmd
echo "cuda_gpu_trace completed."

# Not needed
# cmd="nsys stats -r cuda_kern_exec_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_kern_exec_sum completed."

cmd="nsys stats -r cuda_kern_exec_trace -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
echo $cmd
$cmd
echo "cuda_kern_exec_trace completed."

# Not needed
# cmd="nsys stats -r cuda_api_gpu_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "cuda_api_gpu_sum completed."

echo "All CUDA commands executed successfully. Check the output in ${OUTPUT_DIR}."

# End of script

# Usage
# chmod +x run_nsys_cuda_stats.sh
# ./run_nsys_cuda_stats.sh <output_dir> <input_file> <task_name>
# ./run_nsys_cuda_stats.sh /path/to/output/directory /path/to/nsys_profile.sqlite task_name