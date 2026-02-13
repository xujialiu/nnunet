name_exp="nnunet_chunwen"
num_dataset=101
export nnUNet_results="output/"$name_exp
export nnUNet_raw="/data_B/xujialiu/projects/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data_B/xujialiu/projects/nnunet/nnUNet_preprocessed"
CUDA_VISIBLE_DEVICES=1 python nnUNetv2_train.py $num_dataset 2d 0 --npz

python convert_masks.py output/nnunet_chunwen/Dataset101_chunwen/nnUNetTrainer__nnUNetPlans__2d/fold_0


# 
name_exp="nnunet_chunwen_nnUNetResEncUNetL"
num_dataset=102
export nnUNet_results="output/"$name_exp
export nnUNet_raw="/data_B/xujialiu/projects/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data_B/xujialiu/projects/nnunet/nnUNet_preprocessed"
CUDA_VISIBLE_DEVICES=1 python nnUNetv2_train.py $num_dataset 2d 0 --npz -p nnUNetResEncUNetLPlans

python convert_masks.py output/nnunet_chunwen_nnUNetResEncUNetL/Dataset102_chunwen/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/fold_0