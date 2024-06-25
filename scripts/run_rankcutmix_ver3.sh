CUDA_NUM=1
TRAINER=rankcutmix_ver3
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/
BEST_METRIC_KEY=test_ece
BATCH_SIZE=64

for LAMBDA in 0.2 0.4 0.6 0.8 1.0 
do 
    for MARGIN in 0.05 0.1 0.15 0.2 
    do 
    WANDB_NAME=${DATASET}_${TRAINER}-l${LAMBDA}-m${MARGIN}

    CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
                --dataset=${DATASET} \
                --trainer=${TRAINER} \
                --num_classes=$NUM_CLASSES \
                --basepath=${BASEPATH} \
                --lambd=$LAMBDA \
                --margin=$MARGIN \
                --metric_key=$BEST_METRIC_KEY \
                --batch_size=$BATCH_SIZE \
                --wandb_name=${WANDB_NAME}
    done 
done 
