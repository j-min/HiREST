test_path='./checkpoints/hirest_joint_model/'

echo "Moment Retrieval"
python evaluate.py \
        --task moment_retrieval \
        --pred_data $test_path/test_moment_retrieval_BEST.json

echo '\n'

echo "Moment Segmentation"
python evaluate.py \
        --task moment_segmentation \
        --pred_data $test_path/test_moment_segmentation_BEST.json \
        --preprocess_moment_bounds

echo '\n'

echo "Step Captioning"
python evaluate.py \
        --task step_captioning \
        --pred_data $test_path/test_step_captioning_BEST.json \
        --device 0