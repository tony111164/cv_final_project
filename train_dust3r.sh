export CUDA_VISIBLE_DEVICES=1 # Set the GPU device to use

# torchrun --nproc_per_node=1 train.py \
#     --train_dataset="800 @ SevenScenes(split='train', ROOT='7SCENES', aug_crop=16, resolution=[(256, 192), (256, 160), (256, 128)], transform=ColorJitter)" \
#     --test_dataset="100 @ SevenScenes(split='test', ROOT='7SCENES', resolution=(512,384), seed=777)" \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
#     --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
#     --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
#     --lr=0.0001 --min_lr=1e-06 --warmup_epochs=2 --epochs=5 --batch_size=1 --accum_iter=2 \
#     --save_freq=10 --keep_freq=10 --eval_freq=1 --print_freq=10 \
#     --output_dir="checkpoints/dust3r_7scenes" \
#     --amp 1

torchrun --nproc_per_node=1 train.py \
    --train_dataset="8000 @ SevenScenes(split='train', ROOT='7SCENES', aug_crop=16, resolution=[(256, 192), (256, 160), (256, 128)], transform=ColorJitter)" \
    --test_dataset="100 @ SevenScenes(split='test', ROOT='7SCENES', resolution=(512,384), seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/pretrained_model/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=2 --epochs=10 --batch_size=1 --accum_iter=2 \
    --save_freq=10 --keep_freq=10 --eval_freq=1 --print_freq=10 \
    --output_dir="checkpoints/dust3r_7scenes/train_8000" \
    --amp 1