#!/bin/bash

echo "Pretrain a 3DGS model."

python submodules/gsplat/examples/simple_trainer.py default --data_dir dataset/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/ --data_factor 4 --result_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/pretrained --test_every 1 --disable_viewer

echo "Run Difix."

python examples/gsplat/simple_trainer_difix3d.py default --data_dir dataset/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/ --data_factor 4 --result_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d --test_every 1 --ckpt output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/pretrained/ckpts/ckpt_29999_rank0.pt

echo "Post-rendering"

python src/inference_difix.py --input_image output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d/renders/val/34999/Pred/ --output_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d+/renders/val/34999/Pred --prompt "remove degradation" --timestep 199

python src/inference_difix.py --input_image output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d/renders/val/39999/Pred/ --output_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d+/renders/val/39999/Pred --prompt "remove degradation" --timestep 199

python src/inference_difix.py --input_image output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d/renders/val/44999/Pred/ --output_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d+/renders/val/44999/Pred --prompt "remove degradation" --timestep 199

python src/inference_difix.py --input_image output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d/renders/val/59999/Pred/ --output_dir output/DL3DV-Benchmark-Dreamfix-Split/2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55/difix3d+/renders/val/59999/Pred --prompt "remove degradation" --timestep 199
