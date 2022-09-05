## Eval IJBC

```shell
# model-prefix       your model data_path
# image-data_path         your IJBC data_path
# result-dir         your result data_path
# network            your backbone
CUDA_VISIBLE_DEVICES=0,1 python eval_ijbc.py \
--model-prefix ms1mv3_arcface_r50/backbone.pth \
--image-data_path IJB_release/IJBC \
--result-dir ms1mv3_arcface_r50 \
--batch-size 128 \
--job ms1mv3_arcface_r50 \
--target IJBC \
--network iresnet50
```

## Eval MegaFace

pass

