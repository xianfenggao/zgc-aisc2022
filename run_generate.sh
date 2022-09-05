# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v3.png \
#      --output_dir output/zgc_aisc/generative/ensemble//grid_mask-v3 \
#      --adv_nums 3000 \
#      --src_models TFace_IR_SE_50_2data,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_vggface2,ArcFace_torch_ir100_glint360k \
#      --batch_size 5 \
#      --num_iter 200 \
#      --train_with_eval 

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v4.png \
#      --output_dir output/zgc_aisc/generative/ensemble//grid_mask-v4 \
#      --adv_nums 3000 \
#      --src_models TFace_IR_SE_50_2data,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_vggface2,ArcFace_torch_ir100_glint360k \
#      --batch_size 5 \
#      --num_iter 200 \
#      --train_with_eval 


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v5.png \
#      --output_dir output/zgc_aisc/generative/ensemble//grid_mask-v5 \
#      --adv_nums 3000 \
#      --src_models TFace_IR_SE_50_2data,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_vggface2,ArcFace_torch_ir100_glint360k \
#      --batch_size 5 \
#      --num_iter 200 \
#      --train_with_eval 

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v6.png \
#      --output_dir output/zgc_aisc/generative/ensemble//grid_mask-v6 \
#      --adv_nums 3000 \
#      --src_models TFace_IR_SE_50_2data,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_vggface2,ArcFace_torch_ir100_glint360k \
#      --batch_size 8 \
#      --num_iter 300 \
#      --train_with_eval 

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v7.png \
#      --output_dir output/zgc_aisc/generative/ensemble//grid_mask-v7 \
#      --adv_nums 3000 \
#      --src_models TFace_IR_SE_50_2data,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_vggface2,ArcFace_torch_ir100_glint360k \
#      --batch_size 5 \
#      --num_iter 200 \
#      --train_with_eval 


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/grid_mask-v6.png \
#      --output_dir output/zgc_aisc/generative/ensemble/debug/grid_mask-v6 \
#      --adv_nums 8 \
#      --src_models TFace_IR_SE_50_2data,TFace_MobileFaceNet_Aug_vggface2,TFace_IR101_Aug_glint360k \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia,TFace_DenseNet201_splitbn_faceemore_glintasia \
#      --batch_size 8 \
#      --num_iter 100 \
#      --train_with_eval 

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/rl.png \
#      --output_dir output/zgc_aisc/generative/ensemble/debug/rl \
#      --adv_nums 8 \
#      --src_models TFace_IR_SE_50_2data,TFace_MobileFaceNet_Aug_vggface2,TFace_IR101_Aug_glint360k \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia,TFace_DenseNet201_splitbn_faceemore_glintasia \
#      --batch_size 8 \
#      --num_iter 100 \
#      --train_with_eval 


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main_unique_masks.py \
#      --output_dir output/zgc_aisc/generative/ensemble/unique_masks_facepp_6 \
#      --adv_nums 3000 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 8 \
#      --num_iter 300 \
#      --train_with_eval \
#      --lr 0.01 \
#      --eval_iter 30

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/facepp-lmk.png \
#      --output_dir output/zgc_aisc/generative/ensemble/debug/facepp-lmk \
#      --adv_nums 8 \
#      --src_models TFace_IR_SE_50_2data,TFace_MobileFaceNet_Aug_vggface2,TFace_IR101_Aug_glint360k \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia,TFace_DenseNet201_splitbn_faceemore_glintasia \
#      --batch_size 8 \
#      --num_iter 100 \
#      --train_with_eval 


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/facepp-lmk.png \
#      --output_dir output/zgc_aisc/generative/ensemble/facepp-lmk \
#      --adv_nums 3000 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 8 \
#      --num_iter 300 \
#      --train_with_eval \
#      --lr 0.01 \
#      --eval_iter 30


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask0.png \
#      --output_dir output/zgc_aisc/generative/ensemble/gen_mask/mask0 \
#      --adv_nums 300 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 8 \
#      --num_iter 300 \
#      --train_with_eval \
#      --lr 0.01 \
#      --eval_iter 30

# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask4.png \
#      --output_dir output/zgc_aisc/generative/ensemble/gen_mask/mask4 \
#      --adv_nums 300 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 8 \
#      --num_iter 800 \
#      --train_with_eval \
#      --lr 0.01 \
#      --eval_iter 30


# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask7.png \
#      --output_dir output/zgc_aisc/generative/ensemble/gen_mask/mask7 \
#      --adv_nums 3000 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 8 \
#      --num_iter 800 \
#      --train_with_eval \
#      --lr 0.0888 \
#      --eval_iter 10

# # modifies main.py  add perceptual loss 
python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
     --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask7.png \
     --output_dir output/zgc_aisc/generative/ensemble/gen_mask/mask7-3 \
     --adv_nums 3000 \
     --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
     --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
     --batch_size 6 \
     --num_iter 800 \
     --lr 0.0888 \
     --eval_iter 20 \
     --truncation_psi 0.66

# add input diversity
# python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main.py \
#      --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask7.png \
#      --output_dir output/zgc_aisc/generative/ensemble/gen_mask/mask7-4 \
#      --adv_nums 3000 \
#      --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_IR_SE_50_2data,TFace_IR_SE_50_vggface2 \
#      --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
#      --batch_size 6 \
#      --num_iter 400 \
#      --abort_early 0 \
#      --lr 0.0888 \
#      --eval_iter 20 \
#      --truncation_psi 0.66