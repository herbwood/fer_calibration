CUDA_NUM=2
TRAINER=cutmix
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/


for VERSION in horizontal vertical 
do 
    WANDB_NAME=${DATASET}_${TRAINER}_${VERSION}

    CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
                --dataset=${DATASET} \
                --trainer=${TRAINER} \
                --num_classes=$NUM_CLASSES \
                --basepath=${BASEPATH} \
                --cutmix_ver=${VERSION} \
                --wandb_name=${WANDB_NAME}
done 