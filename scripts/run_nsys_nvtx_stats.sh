#!/bin/bash

# This script is used to get the nsys stats for the given nsys profile file
# Variables
OUTPUT_DIR=$1
INPUT_PATH=$2
TASK_NAME=$3

# Check if the output directory exists, if not create it
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Output directory does not exist"
    mkdir -p $OUTPUT_DIR
fi

# Check if the input file exists
if [ ! -f "$INPUT_PATH" ]; then
  echo "Input file does not exist"
  exit 1
fi

# Get the nsys stats
# cmd="nsys stats -r nvtx_gpu_proj_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "nvtx_gpu_proj_sum completed."

cmd="nsys stats -r nvtx_gpu_proj_trace -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
echo $cmd
$cmd
echo "nvtx_gpu_proj_trace completed."

# cmd="nsys stats -r nvtx_pushpop_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "nvtx_pushpop_sum completed."

cmd="nsys stats -r nvtx_pushpop_trace -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
echo $cmd
$cmd
echo "nvtx_pushpop_trace completed."

# cmd="nsys stats -r nvtx_kern_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "nvtx_kern_sum completed."

# Not needed
# cmd="nsys stats -r nvtx_sum -f csv --output ${OUTPUT_DIR}/${TASK_NAME} ${INPUT_PATH}"
# echo $cmd
# $cmd
# echo "nvtx_sum completed."

# echo "All commands executed successfully. Check the output in ${OUTPUT_DIR}."

# End of script

# Usage
# chmod +x run_nsys_nvtx_stats.sh
# ./run_nsys_nvtx_stats.sh <output_dir> <input_file> <task_name>
# ./run_nsys_nvtx_stats.sh /path/to/output/directory /path/to/nsys_profile.sqlite task_name
