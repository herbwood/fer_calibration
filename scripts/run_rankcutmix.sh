CUDA_NUM=1
TRAINER=rankcutmix
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/
BEST_METRIC_KEY=test_ece
BATCH_SIZE=64

for LAMBDA in 0.2 0.4 0.6 0.8 1.0 
do 
    WANDB_NAME=${DATASET}_${TRAINER}_ver1-l${LAMBDA}

    CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
                --dataset=${DATASET} \
                --trainer=${TRAINER} \
                --num_classes=$NUM_CLASSES \
                --basepath=${BASEPATH} \
                --lambd=$LAMBDA \
                --metric_key=$BEST_METRIC_KEY \
                --batch_size=$BATCH_SIZE \
                --wandb_name=${WANDB_NAME}
done 
