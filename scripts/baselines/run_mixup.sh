CUDA_NUM=1
TRAINER=mixup
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/

for ALPHA in 0.2 0.4 0.6 0.8 1.0
do 
    WANDB_NAME=${DATASET}_${TRAINER}_alpha_${ALPHA}

    CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
                --dataset=${DATASET} \
                --trainer=${TRAINER} \
                --num_classes=$NUM_CLASSES \
                --basepath=${BASEPATH} \
                --alpha=$ALPHA \
                --wandb_name=${WANDB_NAME}
done 