#!/bin/zsh


# Running example:
# ./run_full_model.sh
# ./run_full_model.sh final_results_directory

# Train the model on the full training data
# Iterate ver the dataset {"rest15", "rest16"}
# Iterate over the tasks {"task1", "task2", "task3", "task4", "task5", "mtl"}
# Iterate over the seeds {123, 262, 401, 540, 679}
PASSED_PARAM=$1
# If the passed param is the empty string, use ".". Needed to preserve the logs paths to the current directory
SAVE_DIR=${PASSED_PARAM:-"."}

for dataset in "rest15" "rest16"
do
    for task in "task1" "task2" "task3" "task4" "task5" "mtl"
    do
        for seed in 123 262 401 540 679
        do
            echo """python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --base_path src/weak_supervision_for_few_shot_absa/instruction_tuning --task $task --dataset $dataset --model_name_or_path t5-base --n_gpu 0 --do_train --do_direct_eval --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "${SAVE_DIR}/$dataset/$task" --not_save_checkpoint"""
            # python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --base_path src/weak_supervision_for_few_shot_absa/instruction_tuning --task $task --dataset $dataset --model_name_or_path 'mrm8488/t5-base-finetuned-imdb-sentiment' --n_gpu 0 --do_train --do_direct_eval --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "${SAVE_DIR}/$dataset/$task" --not_save_checkpoint
        done
    done
done