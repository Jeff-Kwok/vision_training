#!/usr/bin/env python3
"""
Train RF-DETR Base on a local dataset (template based on your snippet).

Notes:
- batch_size=2, grad_accum_steps=4 => effective batch size 8
- If you want to start fresh, set resume=None (or remove the arg).
"""

from pathlib import Path
from rfdetr import RFDETRBase


def main():
    DATASET_PATH = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/datasets/training_sets/dataset_8"
    OUTPUT_DIR = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_5"
    RESUME_CKPT = "/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/results/model_5/checkpoint.pth"

    # Basic path sanity checks (optional but helpful)
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"dataset_dir not found: {DATASET_PATH}")

    # If resume checkpoint doesn't exist, fall back to training from pretrained base weights
    if RESUME_CKPT and not Path(RESUME_CKPT).exists():
        print(f"[warn] resume checkpoint not found, training without resume: {RESUME_CKPT}")
        RESUME_CKPT = None

    model = RFDETRBase()

    model.train(
        dataset_dir=DATASET_PATH,
        epochs=80,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-4,
        resolution = 560,
        gradient_checkpointing=True,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.005,
        output_dir=OUTPUT_DIR,
        #resume=RESUME_CKPT,  # set to None to start fresh
    )


if __name__ == "__main__":
    main()