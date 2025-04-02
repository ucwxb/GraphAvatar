time=$(date "+%Y%m%d_%H%M%S")
dataset="insta"
scene="$2"
exp_name="${dataset}_${scene}_full"
dataset_path="./dataset/${dataset}/${scene}"

# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# unset LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 python train.py \
--white_background "white" \
--iterations 100000 \
--lambda_vgg 0.1 \
--eval \
--test_and_save_iterations 10000 \
--model_path "./output/${exp_name}" \
--source_path "${dataset_path}" \
--bs_template_path "${dataset_path}/canonical.obj" \
--warmup_load "${dataset_path}/point_cloud.ply" \
--exp_num 100 --exp_mlp_dim 64 \
--warmup_iteration 30000 \
--lr_rates 0.001 0.0005 0.0005 0.0001 \
--milestones 0 10000 30000 50000 \
--weight_loss --lambda_weight_loss 0.1 \
--pre_loss --lambda_pre_loss 0.1 \
--downsampling_factors 4 8 \
--polygon_order 6 6 6 \
--num_conv_filters 16 16 16 \
--z_dim 8 \
--n_layers 2 \
--post_process \
--metric_xyz --metric_scale \
--enable_scaffold \
--scaffold_feat_dim 4 \
--scaffold_n_offsets 2 \
