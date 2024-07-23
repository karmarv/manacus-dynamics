


## MMYolo - RTMDet
- Python 3.8.18 Installation via Miniconda v23.1.0 - https://docs.conda.io/projects/miniconda/en/latest/
  ```bash
  conda env remove -n mmyolo
  conda create -n mmyolo python=3.9
  conda activate mmyolo
  pip install -r requirements.txt
  ```
- Installation: https://github.com/open-mmlab/mmyolo/blob/main/docs/en/get_started/installation.md 
  ```bash
  pip install -U openmim wandb future tensorboard prettytable
  mim install "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"
  mim install albumentations --no-binary qudida,albumentations
  # Install MMYOLO
  mim install -v -e .
  ```
- Weights and Bias dashboard
  ```bash
  # After running wandb login, enter the API Keys obtained from your project, and the login is successful.
  wandb login 
  ```

### Exp

#### Run - RTMDet-S
```
# 1 GPU # XXX hours for b2 100 epochs
CUDA_VISIBLE_DEVICES=0 PORT=29601 ./tools/dist_train.sh rtmdet_s_manacus.py 1
```
>
```log

```

#### Run - RTMDet-M
