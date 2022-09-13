 
python3 tasks/faceAttacks/PatchAttacks/zgc-aisc/main-final.py \
     --mask_path dataset/Face/zgc-aisc2022/mask/testmask/mask.png \
     --output_dir output/zgc_aisc/final \
     --src_models TFace_IR101_Aug_glint360k,MXNET_LResNet100E_IR_SGM,TFace_DenseNet201_splitbn_faceemore_glintasia,TFace_MobileFaceNet_Aug_faceemore,TFace_MobileFaceNet_Aug_vggface2,TFace_MobileFaceNet_Aug_umd,TFace_MobileFaceNet_Aug_webface \
     --eval_models ArcFace_torch_ir50_glint360k,TFace_IR_SE_50_glintasia,ArcFace_torch_ir50_faceemore,TFace_MobileFaceNet_Aug_glint360k,FRB_MobileNet,FRB_SphereFace,FRB_ResNet50,FRB_ArcFace_IR_50,FaceNet_vggface2,evoLVe_IR_50_Asia \
     --batch_size 6 \
     --num_iter 800 \
     --lr 0.0888 \
     --eval_iter 20 \
     --truncation_psi 0.66
