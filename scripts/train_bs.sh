time=$(date "+%Y%m%d_%H%M%S")
scene="id$2"
dataset="nbs"
exp_name="${dataset}_${scene}_v0519"

CUDA_VISIBLE_DEVICES=$1 python train.py \
--white_background "white" \
--iterations 600000 \
--lambda_vgg 0.1 \
--eval \
--test_and_save_iterations 10000 \
--model_path "./output/${exp_name}" \
--source_path "./nbs-dataset/${scene}" \
--exp_num 120 --exp_mlp_dim 64 \
--warmup_iteration 10000 \
--bs_template_path "./tracker/${scene}/canonical.obj" \
--warmup_load "./warmup_gaussian/nbs_${scene}_warmup/point_cloud/iteration_30000/point_cloud.ply" \
--lr_rates 0.001 0.0005 0.0001 \
--milestones 0 10000 50000 \
--weight_loss --lambda_weight_loss 0.1 \
--pre_loss --lambda_pre_loss 0.1 \
--downsampling_factors 4 8 \
--polygon_order 6 6 6 \
--num_conv_filters 16 16 16 \
--z_dim 8 \
--n_layers 2 \
--metric_xyz --metric_scale \
--post_process \
--use_retrack \
--enable_scaffold \
--scaffold_feat_dim 4 \
--scaffold_n_offsets 2 \
# --dataset_debug \