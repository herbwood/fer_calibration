CUDA_NUM=1
TRAINER=rul
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/
WANDB_NAME={DATASET}_{TRAINER}_ontest

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --wandb_name=${WANDB_NAME}
