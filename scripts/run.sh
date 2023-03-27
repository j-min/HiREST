output='./checkpoints/hirest_joint_model/'
python run.py \
        --data_dir './data/splits/' \
        --video_feature_dir './data/eva_clip_features' \
        --asr_dir './data/ASR' \
        --asr_feature_dir './data/ASR_feats_all-MiniLM-L6-v2' \
        --optim adamw \
        --warmup_steps 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-5 \
        --epochs 50 \
        --num_workers 2 \
        --num_beams 3 \
        --train_batch_size 5 \
        --eval_batch_size 5 \
        --task_moment_retrieval \
        --task_moment_segmentation \
        --task_step_captioning \
        --ckpt_dir $output \
        ${@:1}