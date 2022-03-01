#!/bin/bash

shape=$1 # command line input


datadir='/n/fs/pvl-viewsyn/llff/colmap_dense'
mod='baseline_highres_nobatch'

python run_nerf.py --expname "LLFF_${mod}_${shape}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
--datadir "${datadir}/${shape}" --dataset_type llff \
--use_viewdirs --N_samples 64 --N_importance 128 --N_rand 1024 \
--llffhold 5 --height 600 --raw_noise_std 1e0 \
--N_iters 200000 \
--i_video 1000000 --i_weights 20000 --i_testset 20000 \
--no_batching --render_only
