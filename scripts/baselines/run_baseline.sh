CUDA_NUM=2
BEST_METRIC_KEY=test_ece
DATASET=raf
NUM_CLASSES=7
BASEPATH=/nas_homes/jihyun/RAF_DB/
BEST_METRIC_KEY=test_ece

TRAINER=baseline
WANDB_NAME=${DATASET}_${TRAINER}

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --metric_key=$BEST_METRIC_KEY \
            --wandb_name=${WANDB_NAME}

    
TRAINER=scn
WANDB_NAME=${DATASET}_${TRAINER}

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --metric_key=$BEST_METRIC_KEY \
            --wandb_name=${WANDB_NAME}


TRAINER=rul
WANDB_NAME=${DATASET}_${TRAINER}

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --metric_key=$BEST_METRIC_KEY \
            --wandb_name=${WANDB_NAME}


TRAINER=eac
WANDB_NAME=${DATASET}_${TRAINER}

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --metric_key=$BEST_METRIC_KEY \
            --wandb_name=${WANDB_NAME}


TRAINER=lnsu
WANDB_NAME=${DATASET}_${TRAINER}

CUDA_VISIBLE_DEVICES=$CUDA_NUM python run.py \
            --dataset=${DATASET} \
            --trainer=${TRAINER} \
            --num_classes=$NUM_CLASSES \
            --basepath=${BASEPATH} \
            --metric_key=$BEST_METRIC_KEY \
            --wandb_name=${WANDB_NAME}