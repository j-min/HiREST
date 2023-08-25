import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')

    # Data directories
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--video_feature_dir', type=str, required=True)
    parser.add_argument('--asr_dir', type=str, required=False)
    parser.add_argument('--asr_feature_dir', type=str, required=False)

    # Tasks
    parser.add_argument('--task_moment_retrieval', action='store_true')
    parser.add_argument('--task_moment_segmentation', action='store_true')
    parser.add_argument('--task_step_captioning', action='store_true')

    parser.add_argument('--end_to_end', action='store_true')

    # Training & Optimizer
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    
    # Data loading
    parser.add_argument('--n_model_frames', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)

    # Distributed Training (default: single-gpu)
    parser.add_argument('--distributed', action='store_true')

    # Checkpoints
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/')

    # Model - clip4cap
    parser.add_argument('--num_thread_reader', type=int, default=16)
    parser.add_argument('--n_display', type=int, default=16)
    parser.add_argument('--max_frames_step_captioning', type=int, default=20)
    parser.add_argument('--max_words', type=int, default=48)
    parser.add_argument('--visual_num_hidden_layers', type=int, default=2)
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=2)

    # moment retrieval 
    parser.add_argument('--moment_segmentation_difference_threshold', type=float, default=0.50)
    parser.add_argument('--moment_segmentation_max_iterations', type=int, default=20)

    # step captioning
    parser.add_argument('--num_beams', type=int, default=5)

    # video retrieval
    parser.add_argument('--run_name', type=str, default='clip_g_VR_32frames_avgpool')
    parser.add_argument('--video_retrieval_model', type=str, default='clip_g')
    parser.add_argument('--raw_frame', action='store_true',
                        help='use raw frame instead of video features')
    parser.add_argument('--save_feats', action='store_true',
                        help='save video features to disk')

    return parser
