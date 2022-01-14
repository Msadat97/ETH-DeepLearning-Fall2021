output_path="$SCRATCH/dl-project-outputs"
attribute_path="$output_path/attribute"

latent_path="$output_path/latents/source-ws.npy"
save_path="$output_path/DCI_W"

python DCI.py -latent_path $latent_path -attribute_path $attribute_path -save_path $save_path -mode test

latent_path="$output_path/latents/source-styles.npy"
save_path="$output_path/DCI_S"

python DCI.py -latent_path $latent_path -attribute_path $attribute_path -save_path $save_path -mode test