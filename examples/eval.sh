#!/bin/bash

## https://github.com/LLaVA-VL/LLaVA-NeXT/blob/79ef45a6d8b89b92d7a8525f077c3a3a9894a87d/docs/LLaVA_OneVision.md

cd ..


# MODELNAME="llava-onevision-qwen2-0.5b-ov"

for MODELNAME in llava-onevision-qwen2-0.5b-ov llava-onevision-qwen2-7b-ov 
# for MODELNAME in llava-onevision-qwen2-0.5b-ov
# for MODELNAME in llava-onevision-qwen2-72b-ov-sft
do
MODEL="/ssddata/liuyue/github/VisRAG/pretrained_models/${MODELNAME}"

[[ -d logs/$MODELNAME ]] || mkdir -p logs/$MODELNAME

# image tasks
CFLAGS="-I/ssddata/liuyue/miscellaneous/libaio-0.3.113/usr/include" LDFLAGS="-L/ssddata/liuyue/miscellaneous/libaio-0.3.113/usr/lib" CUTLASS_PATH="/ssddata/liuyue/github/cutlass" \
    accelerate launch --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${MODEL},conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
    --tasks chartqa,docvqa_test,infovqa_test,docvqa_val,infovqa_val \
    --batch_size 1 \
    --verbosity INFO \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs/$MODELNAME \
    2>&1 | tee logs/$MODELNAME/eval.log
done

cd -