# CUDA_NUM=0
# TRAINER=baseline
# DATASET=raf
# NUM_CLASSES=7
# BASEPATH=/nas_homes/jihyun/RAF_DB/
# WANDB_NAME={DATASET}_{TRAINER}_ontest

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
#             --dataset=${DATASET} \
#             --trainer=${TRAINER} \
#             --num_classes=$NUM_CLASSES \
#             --basepath=${BASEPATH} \
#             --wandb_name=${WANDB_NAME}

CUDA_NUM=1
TRAINER=eac
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


# CUDA_NUM=0
# TRAINER=baseline
# DATASET=affectnet
# NUM_CLASSES=8
# BATCH_SIZE=256
# BASEPATH=/nas_homes/jihyun/datasets/AffectNet
# WANDB_NAME={DATASET}_{TRAINER}_ontest

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
#             --dataset=${DATASET} \
#             --trainer=${TRAINER} \
#             --num_classes=$NUM_CLASSES \
#             --basepath=${BASEPATH} \
#             --batch_size=$BATCH_SIZE \
#             --wandb_name=${WANDB_NAME}