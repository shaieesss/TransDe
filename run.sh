export CUDA_VISIBLE_DEVICES=1

"""python main2.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path PSM --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 135
python main2.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path PSM  --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 135

python main2.py --anormly_ratio 0.6 --num_epochs 2   --batch_size 256  --mode train --dataset SMD  --data_path SMD   --input_c 38   --output_c 38  --loss_fuc MSE  --win_size 105  --patch_size 57
python main2.py --anormly_ratio 0.6 --num_epochs 10   --batch_size 256  --mode test    --dataset SMD   --data_path SMD     --input_c 38      --output_c 38   --loss_fuc MSE   --win_size 105  --patch_size 57

python main2.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset SWAT  --data_path SWAT  --input_c 51    --output_c 51  --loss_fuc MSE --patch_size 357 --win_size 105
python main2.py --anormly_ratio 1  --num_epochs 10   --batch_size 128     --mode test    --dataset SWAT   --data_path SWAT  --input_c 51    --output_c 51   --loss_fuc MSE --patch_size 357 --win_size 105

python main2.py --anormly_ratio 1 --num_epochs 3   --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 90  --patch_size 35
python main2.py --anormly_ratio 1  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 90  --patch_size 35

python main2.py --anormly_ratio 0.9 --num_epochs 3   --batch_size 128  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 36  --patch_size 13
python main2.py --anormly_ratio 0.9  --num_epochs 10     --batch_size 128   --mode test    --dataset NIPS_TS_Swan   --data_path NIPS_TS_Swan --input_c 38    --output_c 38    --loss_fuc MSE       --win_size 36   --patch_size 13

python main2.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Water  --data_path NIPS_TS_Water  --input_c 9 --output_c 9  --loss_fuc MSE   --patch_size 135  --win_size 90
python main2.py --anormly_ratio 1  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Water   --data_path NIPS_TS_Water --input_c 9    --output_c 9    --loss_fuc MSE   --patch_size 135   --win_size 90
"""


python main2.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 128  --mode train --dataset UCR_AUG  --data_path UCR_AUG   --input_c 1 --output 1 --index 1 --win_size 60  --patch_size 35
python main2.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 128    --mode test    --dataset UCR_AUG   --data_path UCR_AUG     --input_c 1   --output 1  --index 1  --win_size 60 --patch_size 35


"""
for i in {1..250};
do

python main2.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 128  --mode train --dataset UCR  --data_path UCR   --input_c 1 --output 1 --index $i --win_size 105 --patch_size 357
python main2.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 128    --mode test    --dataset UCR   --data_path UCR     --input_c 1   --output 1  --index $i --win_size 105 --patch_size 357

done"""