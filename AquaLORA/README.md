We modify the code from AquaLoRA.

# AquaLoRA: Toward White-box Protection for Customized Stable Diffusion Models via Watermark LoRA

## Quickstart

Run the following command to install the environment:

```bash
git clone https://github.com/Georgefwt/AquaLoRA.git
cd AquaLoRA
conda create -n aqualora python=3.10
conda activate aqualora
pip install -r requirements.txt
```

## Training

See `pretrain_detector.sh` for details on pre-training AquaLoRA detector.