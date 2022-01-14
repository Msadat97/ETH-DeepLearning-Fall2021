python align_mask.py  -gradient_folder "$SCRATCH/dl-project-outputs/grads-correct-shape.pkl" -semantic_path "$SCRATCH/dl-project-outputs/semantic_mask.npy" -save_folder "$SCRATCH/dl-project-outputs//align_mask_32" -num_per 104

python semantic_channel.py -align_folder "$SCRATCH/dl-project-outputs/align_mask_32" -save_folder  "$SCRATCH/dl-project-outputs" 




















