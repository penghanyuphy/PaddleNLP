gpus=$1
python -m paddle.distributed.launch --gpus "$gpus" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path ~/.paddlenlp/models/facebook/opt-350m \
    --input_dir "../../openwebtext/"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 5000\
    --save_steps 1000\
    --decay_steps 3200\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
