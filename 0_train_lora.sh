#!/bin/bash

# This script calls the s1_peft LoRA training script

# Change to the s1_peft directory and run the training script
cd s1_peft && bash train/sft_lora.sh "$@"