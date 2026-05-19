#!/bin/bash

oft_block_size=2
butterfly_factor=2
lr=5e-5
subjects=(
    'backpack' 'backpack_dog' 'bear_plushie' 'berry_bowl' 'can'
    'candle' 'cat' 'cat2' 'clock' 'colorful_sneaker'
    'dog' 'dog2' 'dog3' 'dog5' 'dog6'
    'dog7' 'dog8' 'duck_toy' 'fancy_boot' 'grey_sloth_plushie'
    'monster_toy' 'pink_sunglasses' 'poop_emoji' 'rc_car' 'red_cartoon'
    'robot_toy' 'teapot' 'vase' 'wolf_plushie' 'shiny_sneaker'
)

for subject in "${subjects[@]}"; do
    export CONF_DIR="dataset/Customization/${subject}/config.json"
    export EXP_NAME="block${oft_block_size}_level${butterfly_factor}_lr${lr}/${subject}"

    accelerate launch run_boft.py \
        --config_dir=$CONF_DIR \
        --config_name="gpt_cc" \
        --output_dir="./output" \
        --group_name="boft" \
        --exp_name=$EXP_NAME \
        --learning_rate=$lr \
        --dcoloss_beta=1000 \
        --max_train_steps=2000 \
        --checkpointing_steps=1000 \
        --seed="0" \
        --num_validation_images=4 \
        --validation_epochs=200 \
        --validation_starting_epochs=1 \
        --oft_block_size=$oft_block_size \
        --butterfly_factor=$butterfly_factor

    export LOG_DIR="output/boft/block${oft_block_size}_level${butterfly_factor}_lr${lr}/"
    python evaluate.py \
        --subject=$subject \
        --image_dir=$LOG_DIR
done