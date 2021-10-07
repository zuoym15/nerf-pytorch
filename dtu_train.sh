light_number=3
# shape='scan113'
shape='scan76'
#shape='scan70'


python run_nerf.py --expname "DTU_invpose_${shape}_${light_number}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
--datadir /n/fs/pvl-mvs/DTU_HR/train --dataset_type dtu \
--use_viewdirs --lrate_decay 500 --N_samples 64 --N_importance 128 --N_rand 1024 \
--precrop_iters 500 --precrop_frac 0.5 --i_testset 10000 \
--scan_name ${shape} --light_number ${light_number} \
--N_iters 400000
# --no_batching