from pathlib import Path

training_configs = dict(
    default=dict(
        loss="L2",
        optim="adam",
        learning_rate=0.0001,
        batch_size=16,
        sigma_jitter=0.05,
        max_epochs=300,
        gpus=1,
        nan_value_target=-1000,
    ),
    debug=dict(
        loss="L2",
        optim="adam",
        learning_rate=0.0001,
        batch_size=16,
        sigma_jitter=0.05,
        max_epochs=1,
        gpus=1,
        nan_value_target=-1000,
    ),
)


datasplit_configs = dict(
    uniformly_rdm=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/uniformly-rdm-split.json"
        ),
    ),
    rdm_spatial=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/rdm-spatial-split.json"
        ),
    ),
    rdm_temporal=dict(
        split_mode=str(
            Path(__file__).resolve().parent.parent / "splits/rdm-temporal-split.json"
        ),
    ),
    structured_temporal=dict(
        split_mode="structured", train_years_to=2002, val_years_to=2012
    ),
)

model_configs = dict(
    phenoformer_default=dict(
        n_layers=1,
        nhead=8,
        d_model=64,
        dim_feedforward=128,
    ),
    phenoformer_staticdata=dict(
        n_layers=1,
        nhead=8,
        d_model=64,
        dim_feedforward=128,
        elevation=True,
        latlon=False,
    ),
)
