path_Heart_Train="/Users/gemwincanete/Audio/denoise/Data/PHS Data (Processed)/train"
path_Lung_Train="/Users/gemwincanete/Audio/denoise/Data/ICBHI Dataset (Processed)/train"
pathheartVal='/Users/gemwincanete/Audio/denoise/Data/OAHS Dataset/Git_val'
pathlungval='/Users/gemwincanete/Audio/denoise/Data/ICBHI Dataset (Processed)/val'
pathhospitalval='/Users/gemwincanete/Audio/denoise/Data/Hospital Ambient Noise (HAN) Dataset'
pathPascal = "/Users/gemwincanete/Audio/denoise/Data/PHS Data (Processed)/val"
#pathPascal='../input/clear-heart-data/Processed_6/Processed_6/raw_2400_val'

window_size=.8
name_model="lunet"
input_shape=800
output_shape=800
sampling_rate_new=1000

#check=rf"/content/drive/MyDrive/Heart Sound Denoising All Data/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB.h5"
#check=rf"./check_points/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB.h5"
