conda run -n nertag_env --no-capture-output \
    python train.py --runno 6 \
        --model_type lstm \
        --max_epochs 100 \
        --base_lr 0.001 \
        --verbose \
        --num_decoder_layers 12 \
        --num_encoder_layers 12 \
        --num_dense_layers 7 \
        --d_model 512 \
        --detect_anomaly \
        --enable_autocast \
        --layer_norm_eps 0.01 \
        --batch_size 32 \
        --lstm_input_size 128