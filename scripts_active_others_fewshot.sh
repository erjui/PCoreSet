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


datasets=(
    "caltech101"
    # "dtd"
    # "eurosat"
    # "fgvc"
    # "food101"
    # "oxford_flowers"
    # "oxford_pets"
    # "stanford_cars"
    # "sun397"
    # "ucf101"
)

num_classes=(
    100
    # 47
    # 10
    # 100
    # 101
    # 102
    # 37
    # 196
    # 397
    # 101
)

strategies=(
    "random"
    "coreset"
    "uncertainty"
    "badge"
    "classbalanced"
    "pcoreset"
)

seeds=(
    1
    # 2
    # 3
    # 4
    # 5
)

# Initialize GPU settings
available_gpus=(0 1 2 3 4 5 6 7)
available_gpusgpu_index=0
experiments_per_gpu=1
current_gpu_count=0
active_pids=()  # Add array to track active PIDs

for dataset in "${datasets[@]}"; do
    # Get the index of the current dataset
    index=0
    for i in "${!datasets[@]}"; do
        if [[ "${datasets[$i]}" = "${dataset}" ]]; then
            index=$i
            break
        fi
    done

    # Get the number of classes for the current dataset
    queries=${num_classes[$index]}

    for strategy in "${strategies[@]}"; do
        for seed in "${seeds[@]}"; do
            # Create logs directory for dataset and seed if it doesn't exist
            mkdir -p "./logs/fewshot_others/${dataset}/${seed}"

            # Define log file path
            log_file="./logs/fewshot_others/${dataset}/${seed}/dho_res18_zs_1shot_active_${strategy}_q${queries}_e200_r15_s${seed}.log"

            # Check if log file exists
            if [ -f "$log_file" ]; then
                echo "Skipping: Log file already exists for dataset: $dataset, strategy: $strategy, seed: $seed"
                continue
            fi

            current_gpu=${available_gpus[$gpu_index]}
            echo "Training with strategy: $strategy on dataset: $dataset using GPU $current_gpu with seed $seed"

            CUDA_VISIBLE_DEVICES=$current_gpu python train_active_others.py \
                --dataset $dataset \
                --shots 1 \
                --student_model res18 \
                --batch_size 64 \
                --train_epoch 200 \
                --lr 0.001 \
                --root_path ./data \
                --log_dir "./logs/fewshot_others/${dataset}/${seed}/" \
                --active_learning_rounds 15 \
                --active_learning_strategy $strategy \
                --active_learning_queries $queries \
                --seed $seed \
                --teacher_type clap \
                > "$log_file" 2>&1 &

            # Store the PID of the background process
            active_pids+=($!)

            # Increment experiment counter for current GPU
            current_gpu_count=$((current_gpu_count + 1))

            # If we've run enough experiments on current GPU, move to next GPU and reset counter
            if [ $current_gpu_count -eq $experiments_per_gpu ]; then
                gpu_index=$(( (gpu_index + 1) % ${#available_gpus[@]} ))
                current_gpu_count=0

                # If we've used all GPUs, wait for the current batch to complete
                if [ $gpu_index -eq 0 ]; then
                    echo "Waiting for current batch of GPU jobs to complete..."
                    wait "${active_pids[@]}"
                    active_pids=()
                fi
            fi
        done
    done
done

# Wait for any remaining background processes to complete
echo "Waiting for all remaining jobs to complete..."
wait
