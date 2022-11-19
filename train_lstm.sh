conda run -n nertag_env --no-capture-output \
    python train.py --runno 3 \
        --model_type lstm \
        --max_epochs 100 \
        --base_lr 0.005 \
        --verbose \
        --num_decoder_layers 15 \
        --num_encoder_layers 15 \
        --num_dense_layers 7 \
        --d_model 512 \
        --detect_anomaly \
        --enable_autocast \
        --layer_norm_eps 0.01 \
        --batch_size 256 \
        --lstm_input_size 64