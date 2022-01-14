cd ./face_parsing/
python GetMask2.py -model_path $SCRATCH/79999_iter.pth -img_path $SCRATCH/dl-project-outputs/stylegan3-fake-images.npy  -save_path $SCRATCH/dl-project-outputs/semantic_mask.npy