shape='fern' # bird

# datadir='/n/fs/pvl-viewsyn/llff/nerf_llff_data'
datadir='/n/fs/pvl-viewsyn/llff/colmap_dense'

mod='baseline_lowres_nobatch'

python run_nerf.py --expname "LLFF_${mod}_${shape}" --basedir /n/fs/pvl-viewsyn/nerf/logs \
--datadir "${datadir}/${shape}" --dataset_type llff \
--use_viewdirs --N_samples 64 --N_importance 64 --N_rand 1024 \
--llffhold 5 --height 300 --raw_noise_std 1e0 \
--N_iters 200000 \
--i_video 1000000 --i_weights 50000 --i_testset 50000 \
--no_batching
