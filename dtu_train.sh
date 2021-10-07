light_number=3

# shape='scan113' # golden rabbit
# shape='scan76' # veges
#shape='scan70' # snowman
shape='scan4' # bird

# datadir='/n/fs/pvl-mvs/DTU_HR/train'
datadir='/u/zeyum/x/DTU/mvs_testing'


python run_nerf.py --expname "DTU_invpose_${shape}_${light_number}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
--datadir ${datadir} --dataset_type dtu \
--use_viewdirs --lrate_decay 500 --N_samples 64 --N_importance 128 --N_rand 1024 \
--precrop_iters 500 --precrop_frac 0.5 \
--scan_name ${shape} --light_number ${light_number} \
--N_iters 400000 \
--i_video 1000000 --i_weights 50000 \
# --no_batching