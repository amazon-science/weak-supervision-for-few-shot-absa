#!/bin/zsh

PASSED_PARAM=$1
# If the passed param is the empty string, use ".". Needed to preserve the logs paths to the current directory
SAVE_DIR=${PASSED_PARAM:-"."}

for k in 5 10
do
    for dataset in "lap14"
    do
        for task in "task1" "task2" "task4" # "mtl"
        do
            for k_seed in 12347
            do
                for seed in 1 2 3 4 5
                do
                    # echo """python -m src.weak_supervision_for_few_shot_absa.in[struction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 2 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar """
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 2 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar # --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 2 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/pft/fewshot/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt"
                done
            done
        done
    done
done

for k in 20
do
    for dataset in "lap14"
    do
        for task in "task1" "task2" "task4" # "mtl"
        do
            for k_seed in 12347
            do
                for seed in 1 2 3 4 5
                do
                    # echo """python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week9/pft/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" """
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar # --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/pft/fewshot/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                done
            done
        done
    done
done

for k in 50
do
    for dataset in "lap14"
    do
        for task in "task1" "task2" "task4" # "mtl"
        do
            for k_seed in 12347
            do
                for seed in 1 2 3 4 5
                do
                    # echo """python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week9/pft/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" """
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar # --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/pft/fewshot/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                done
            done
        done
    done
done






#####################################################################################




for k in 20
do
    for dataset in "rest15"
    do
        for task in "task1" "task2" "task3" "task4" "task5" # "mtl"
        do
            for k_seed in 12347
            do
                for seed in 1 2 3 4 5
                do
                    # echo """python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week9/pft/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" """
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar # --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/pft/fewshot/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                done
            done
        done
    done
done

for k in 50
do
    for dataset in "rest15"
    do
        for task in "task1" "task2" "task3" "task4" "task5" # "mtl"
        do
            for k_seed in 12347
            do
                for seed in 1 2 3 4 5
                do
                    # echo """python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week9/pft/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" """
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/original/fs_${k}/${dataset}/${task}"    --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar # --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                    python -m src.weak_supervision_for_few_shot_absa.instruction_tuning.main --task "${task}" --train_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/train.jsonl" --val_dataset_path "data_as_jsonl/${dataset}/fewshot/${dataset}_${k}/dev.jsonl" --evaluation_dataset_path_dev "data_as_jsonl/${dataset}/dev.jsonl" --template_name "main_templates" --generated_sequence_dropout_probability 0.0 --model_name_or_path t5-base --n_gpu 0 --do_train --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 16 --reload_dataloaders_every_epoch --learning_rate 3e-4 --num_train_epochs 20 --seed $seed --prefetch_factor 20 --log_save_name "week10/fewshot/pft/fewshot/fs_${k}/${dataset}/${task}" --do_direct_eval --not_save_checkpoint --keep_last_batch --enable_progress_bar --checkpoint_path "logs/week10/yelp_reviews_linked/ate_ote/reg_hs_search/version_34/checkpoints/last.ckpt" 
                done
            done
        done
    done
done


