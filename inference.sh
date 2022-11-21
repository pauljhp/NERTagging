conda run -n nertag_env --no-capture-output \
    python inference.py \
        --verbose \
        --nhead 16 \
        --no_dense_layers 7 \
        --d_model 256 \
        --detect_anomaly \
        --enable_autocast \
        --layer_norm_eps 0.01 \
        --batch_size 256