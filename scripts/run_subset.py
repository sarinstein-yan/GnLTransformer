import gnl_transformer.models as M
from gnl_transformer.training import run_experiment

gnl = M.GnLTransformer(
    dim_in_G=2,
    dim_in_L=11,
    dim_h_conv=32,
    dim_h_lin=512,
    dim_out=36,
    num_layer_conv=4,
    num_layer_lin=2,
    num_heads=4,
    pool_k_G=30,
    pool_k_L=30,
    dropout=0.,
)

train_hp = {
    "root": "/home/user/research/GnLTransformer/dev",
    "save_dir": "/home/user/research/GnLTransformer/dev/baseline_result",
    "epochs": 200,
    "batch_size": 4096,
    "seeds": [42, 2025, 1, 95, 27],
    # optimizer and scheduler
    "lr_init": 1e-3,
    "lr_min": 1e-5,
    "t0": 200,
    "t_mult": 2,
    # callbacks
    "log_every_n_steps": 10,
    "early_stop_patience": 200,
}

run_experiment(gnl, train_hp)