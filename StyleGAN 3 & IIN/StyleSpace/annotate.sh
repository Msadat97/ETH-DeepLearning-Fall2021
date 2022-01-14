output_path="$SCRATCH/dl-project-outputs"
img_path="$output_path/stylegan3-fake-images.npy"
save_path="$output_path/attribute"
classifer_path="$output_path/checkpoints"
python GetAttribute.py -img_path  $img_path -save_path $save_path -classifer_path $classifer_path

python GetMask2.py -model_path $SCRATCH/79999_iter.pth -img_path $SCRATCH/dl-project-outputs/stylegan3-fake-images.npy -save_path $SCRATCH/dl-project-outputs/semantic_mask.npy
