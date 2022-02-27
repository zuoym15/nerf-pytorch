#!/bin/bash

light_number=3

dataset=$1 # command line input
shape=$2 # command line input

if [ "$dataset" == 'LLFF' ]
then
    datadir='/n/fs/pvl-viewsyn/llff/colmap_dense'
    # mod='baseline_lowres_batching'
    mod='baseline_highres_nobatch'
    #mod='baseline_4096_highres_batching'

    python run_nerf.py --expname "LLFF_${mod}_${shape}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
    --datadir "${datadir}/${shape}" --dataset_type llff \
    --use_viewdirs --N_samples 64 --N_importance 128 --N_rand 1024 \
    --llffhold 5 --height 600 --raw_noise_std 1e0 \
    --N_iters 200000 \
    --i_video 1000000 --i_weights 20000 --i_testset 20000 \
    --no_batching

elif [ "$dataset" == 'DTUHR' ]
then
    datadir='/u/zeyum/x/DTU/mvs_testing'
    mod='dtu_all'

    python run_nerf.py --expname "DTU_${mod}_${shape}_${light_number}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
    --datadir ${datadir} --dataset_type dtu \
    --use_viewdirs --lrate_decay 500 --N_samples 64 --N_importance 128 --N_rand 2048 \
    --scan_name ${shape} --light_number ${light_number} \
    --N_iters 200000 \
    --i_video 1000000 --i_weights 50000 --i_testset 50000 \
    --batching_end_iter 1000

else
   echo "unsupported dataset"
   exit 1
fi


# highres-no batch
#python run_nerf.py --expname "LLFF_${mod}_${shape}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
#    --datadir "${datadir}/${shape}" --dataset_type llff \
#    --use_viewdirs --N_samples 64 --N_importance 128 --N_rand 1024 \
#    --llffhold 5 --height 600 --raw_noise_std 1e0 \
#    --N_iters 200000 \
#    --i_video 1000000 --i_weights 20000 --i_testset 20000 \
#    --no_batching