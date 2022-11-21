conda run -n nertag_env --no-capture-output \
    python train.py --runno 2 \
        --model_type transformer \
        --max_epochs 100 \
        --base_lr 0.0005 \
        --embedding_type glove \
        --freeze_embedding_params \
        --verbose \
        --nhead 25 \
        --num_dense_layers 7 \
        --num_decoder_layers 10 \
        --num_encoder_layers 10 \
        --d_model 300 \
        --detect_anomaly \
        --enable_autocast \
        --layer_norm_eps 0.01 \
        --batch_size 256