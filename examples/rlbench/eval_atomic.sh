# === Evaluation Tasks ===
root_tasks=(
    "box_in_cupboard"
    "box_out_of_opened_drawer"
    "close_drawer"
    "put_in_opened_drawer"
    "sweep_to_dustpan"
    "box_out_of_cupboard"
    "broom_out_of_cupboard"
    "open_drawer"
    "rubbish_in_dustpan"
    "take_out_of_opened_drawer"
)
data_dir=/data1/cyt/HiMan_data/test_atomic

# === Evaluation Loop ===
for root_task in "${root_tasks[@]}"; do
    for i in {0..17}; do
        task_name="${root_task}_${i}"
        DATA_PATH="${data_dir}/${task_name}/"

        if [ ! -d "$DATA_PATH" ]; then
            echo "[Skip] $DATA_PATH does not exist."
            continue
        fi

        python examples/rlbench/main.py \
            --args.tasks="$task_name" \
            --args.host=114.212.189.99 \
            --args.port=8001 \
            --args.max_steps=20 \
            --args.data_dir=$data_dir \
            --args.num_episodes=1 \
            --args.tasks_type=atomic \
            --args.output_file=/data1/cyt/HiMan_data/train_logs_baseline_openpi/openpi_baseline_low_mem_finetune/my_experiment/eval_log_atomic
    done
done