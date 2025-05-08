import os

input_folder = '..//000_datasets//Lung_CT'

for i in range(1, 102, 99):   # (1): #
    os.system('python main.py --ni --config imagenet_256.yml --output_path results_Lung_CT \
    --path_y celeba_hq --eta 0.85 --deg "denoising" --output_folder Lung_CT_SR \
    --deg_scale 1.0 --noise_rate 0 --add_noise \
    --lambda_t '+str(i)+' --input_folder '+input_folder)
