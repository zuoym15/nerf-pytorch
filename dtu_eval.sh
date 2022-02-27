light_number=3

# shape='scan113'
# shape='scan76'
# shape='scan70'
# shape='scan4' # bird
# shape='scan63' # fruits
shape=$1

# datadir='/n/fs/pvl-mvs/DTU_HR/train'
datadir='/u/zeyum/x/DTU/mvs_testing'

mod='dtu_all'


python run_nerf.py --expname "DTU_${mod}_${shape}_${light_number}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
--datadir ${datadir} --dataset_type dtu \
--use_viewdirs --lrate_decay 500 --N_samples 64 --N_importance 128 --N_rand 1024 \
--precrop_iters 500 --precrop_frac 0.5 --i_testset 10000 \
--scan_name ${shape} --light_number ${light_number} \
--render_only

# --render_test --val_cam_noise 0.005

# --foreground_mask_path /n/fs/pvl-viewsyn/dtu_mask