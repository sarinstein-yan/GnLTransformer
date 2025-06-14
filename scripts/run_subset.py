import gnl_transformer.models as M
from gnl_transformer.training import run_experiment

train_hp = {
    "root": "/home/user/research/GnLTransformer/dev",
    "save_dir": "/home/user/research/GnLTransformer/dev/baseline_result",
    "epochs": 200,
    "batch_size": 1024,
    "seeds": [42, 2025, 1, 95, 27, 45, 1024],
    "subset_ratio": 0.2,
    # optimizer and scheduler
    "lr_init": 1e-3,
    "lr_min": 1e-5,
    "t0": 200,
    "t_mult": 2,
    # callbacks
    "log_every_n_steps": 10,
    "early_stop_patience": 200,
}

gnl = M.GnLTransformer(
    dim_in_G=2,
    dim_in_L=11,
    dim_h_conv=32,
    dim_h_lin=512,
    dim_out=36,
    num_layers_conv=4,
    num_layers_lin=2,
    num_heads=4,
    pool_k_G=30,
    pool_k_L=30,
    dropout=0.,
)

gnl_ablt = M.GnLTransformer_ablation(
    dim_in_G=2,
    dim_h_conv=32,
    dim_h_lin=512,
    dim_out=36,
    num_layers_conv=4,
    num_layers_lin=2,
    num_heads=4,
    pool_k_G=30,
    dropout=0.,
)


shared_hps = {
    "dim_in": 2,
    "dim_out": 36,
    "num_layers_conv": 4,
    "num_layers_lin": 2,
    "dropout": 0.,
}

gcn = M.BasicGNNBaselines(
    M.GCN, **shared_hps,
    dim_h_conv=480, dim_h_lin=950,
)

gin = M.BasicGNNBaselines(
    M.GIN, **shared_hps,
    dim_h_conv=350, dim_h_lin=700,
)

gine = M.BasicGNNBaselines(
    M.GINE, **shared_hps,
    dim_h_conv=350, dim_h_lin=700,
)

gat = M.BasicGNNBaselines(
    M.GAT, **shared_hps,
    dim_h_conv=460, dim_h_lin=920,
)

gatv2 = M.BasicGNNBaselines(
    M.GAT, **shared_hps,
    v2=True,
    dim_h_conv=370, dim_h_lin=740,
)
gatv2.alias = "GATv2"
gatv2.hps['alias'] = "GATv2"

mf = M.MF(**shared_hps,
    dim_h_conv=125, dim_h_lin=250,
)

afp = M.AFP(**shared_hps, 
    edge_dim=11, 
    num_timesteps=2,
    dim_h_conv=170, dim_h_lin=340,
)

models = [
    gnl, gnl_ablt,
    gcn, gin, gine, gat, gatv2, mf, afp,
]


for model in models:
    run_experiment(model, train_hp)