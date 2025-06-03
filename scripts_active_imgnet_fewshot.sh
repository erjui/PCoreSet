# Function to handle Ctrl+C
handle_interrupt() {
    echo -e "\n\nReceived interrupt signal. Cleaning up..."
    # Kill all background processes
    jobs -p | xargs -r kill
    echo "All training processes have been terminated."
    exit 1
}

# Set up trap for SIGINT (Ctrl+C)
trap handle_interrupt SIGINT

# Define fixed parameters
dataset="imagenet"
num_classes=1000
# gpu="0,1,2,3"
gpu="4,5,6,7"
num_gpus=4
seed=1
max_query_samples=100000

strategies=(
    "random"
    "uncertainty"
    "coreset"
    "classbalanced"
    "pcoreset"
)

# Create logs directory if it doesn't exist
mkdir -p "./logs/fewshot_imgnet/${dataset}/seed${seed}"

# Loop through each strategy
for strategy in "${strategies[@]}"; do
    # Define log file path
    log_file="./logs/fewshot_imgnet/${dataset}/seed${seed}/dho_res50_zs_1shot_active_${strategy}_q${num_classes}_e20_r8.log"

    # Check if log file exists
    if [ -f "$log_file" ]; then
        echo "Skipping: Log file already exists for dataset: $dataset, strategy: $strategy, seed: $seed"
        continue
    fi

    echo "Training with strategy: $strategy on dataset: $dataset using DDP with $num_gpus GPU(s), seed: $seed"

    # Use torch.distributed.launch for DDP training
    CUDA_VISIBLE_DEVICES=$gpu NCCL_TIMEOUT=1800000 python -m torch.distributed.launch \
        --nproc_per_node=$num_gpus \
        --master_port=29514 \
        train_active_imgnet.py \
        --dataset $dataset \
        --shots 1 \
        --teacher_type clap \
        --batch_size 256 \
        --train_epoch 1 \
        --lr 0.001 \
        --root_path ./data \
        --log_dir ./logs/fewshot_imgnet/ \
        --active_learning \
        --active_learning_rounds 2 \
        --active_learning_strategy $strategy \
        --active_learning_queries $num_classes \
        --max_query_samples $max_query_samples \
        --seed $seed \
        > "$log_file" 2>&1

    # # Use torch.distributed.launch for DDP training
    # CUDA_VISIBLE_DEVICES=$gpu NCCL_TIMEOUT=1800000 python -m torch.distributed.launch \
    #     --nproc_per_node=$num_gpus \
    #     --master_port=29514 \
    #     train_active_imgnet.py \
    #     --dataset $dataset \
    #     --shots 1 \
    #     --teacher_type clap \
    #     --batch_size 256 \
    #     --train_epoch 20 \
    #     --lr 0.001 \
    #     --root_path ./data \
    #     --log_dir ./logs/fewshot_imgnet/ \
    #     --active_learning \
    #     --active_learning_rounds 8 \
    #     --active_learning_strategy $strategy \
    #     --active_learning_queries $num_classes \
    #     --max_query_samples $max_query_samples \
    #     --seed $seed \
    #     > "$log_file" 2>&1

    echo "Training job for strategy $strategy completed!"
done

echo "All training jobs completed!"
