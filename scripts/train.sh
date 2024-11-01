for interlingua_length in 28
do
for u2c_temp in 1.0
do
u2u_temp=1.0
query_update='cl'
for u2u_weight in 1.0
do
u2c_weight=${u2u_weight}
contra_weight=1.0
for loss in tl_euc_gan_lt2m_lm2t_l2m_l2t
do
home_dir=.
data_dir=./data/text2mol
model=GCN_GAN_QUERY
ma_type=query
gan_weight=2e-4
model_name=${model}_${loss}_q${interlingua_length}_u2ut${u2u_temp}_u2uw${u2u_weight}_u2ct${u2c_temp}_u2cw${u2c_weight}_contraw${contra_weight}_${ma_type}_gan${gan_weight}
batch_size=32
epochs=60
wandb_project=wandb
wandb_name=$model_name
gpu_id=1
output_path=$home_dir/output
CUDA_VISIBLE_DEVICES=$gpu_id python $home_dir/code/main.py \
        --data $data_dir --home-dir $home_dir \
        --model $model --epochs $epochs --batch_size $batch_size \
        --model-name $model_name \
        --loss $loss \
        --ma-type $ma_type --gan-weight $gan_weight \
        --query-update $query_update \
        --interlingua-length $interlingua_length \
        --u2u-temp $u2u_temp --u2c-temp $u2c_temp \
        --u2u-weight $u2u_weight --u2c-weight $u2c_weight \
        --contra-weight $contra_weight \
        --wandb --wandb-project $wandb_project --wandb-name $wandb_name
done
done
done
done