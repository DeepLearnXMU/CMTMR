query_update=cl
loss=tl_euc_gan_u2ueuc_u2ceuc
home_dir=.
model=GCN_GAN_QUERY
model_name=finetune_para
batch_size=64
epochs=30
wandb_project=test
wandb_name=$model_name
gan_weight=2e-4
gpu_id=4
ma_type=query
ckpt=GCN_GAN_QUERY_tl_euc_gan_lt2m_lm2t_l2m_l2t_q28_u2ut1.0_u2uw1.0_u2ct1.0_u2cw1.0_contraw1.0_query_gan2e-4
train_from=$home_dir/output/${ckpt}/models/checkpoint_last.pt
output_path=$home_dir/output
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$gpu_id python $home_dir/code/finetune.py \
        --home-dir $home_dir \
        --datatype 0 \
        --bert_lr 5e-5 \
        --model $model --epochs $epochs --batch_size $batch_size \
        --model-name $model_name \
        --loss $loss \
        --ma-type $ma_type --gan-weight $gan_weight \
        --query-update $query_update \
        --interlingua-length 28 --train-from $train_from \
        --u2u-temp 1 --u2c-temp 1 \
        --u2u-weight 1.0 --u2c-weight 1.0 \
        --contra-weight 1.0 --seed 42 \
        --wandb --wandb-project $wandb_project --wandb-name $wandb_name \