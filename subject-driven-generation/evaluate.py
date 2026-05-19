#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel, AutoImageProcessor, AutoModel

import lpips
import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
import re

CLASS_DICT = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal",
}


def get_prompt(subject_name, prompt_idx):
    config_dir = os.path.join('dataset/Customization', subject_name, 'config.json')
    with open(config_dir, 'r') as data_config:
        data_cfg = json.load(data_config)["gpt_cc"]
    
    # return data_cfg["eval_prompts"][int(prompt_idx)]
    return data_cfg["eval_prompts"][int(prompt_idx)].replace(subject_name, CLASS_DICT[subject_name])


class PromptDatasetSigLIP(Dataset):
    def __init__(
            self, subject_name, data_dir_B, processor,
            iteration, prompt_id
        ):
        self.data_dir_B = data_dir_B
            
        # subject_name, prompt_idx = subject_name.split('-')
        
        # data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        # self.image_lst = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        self.image_lst = [
            os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B)
            if f.endswith(".png") and f.startswith(f'{str(iteration)}_{str(prompt_id)}')
        ]
        self.prompt_lst = [get_prompt(subject_name, prompt_id)] * len(self.image_lst)
        
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            inputs = self.processor(text=[prompt], images=image, padding="max_length", return_tensors="pt")
            return inputs


class PromptDatasetCLIP(Dataset):
    def __init__(
            self, subject_name, data_dir_B, tokenizer, processor,
            iteration, prompt_id
        ):
        self.data_dir_B = data_dir_B
            
        # subject_name, prompt_idx = subject_name.split('-')
        
        # data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        # self.image_lst = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        self.image_lst = [
            os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B)
            if f.endswith(".png") and f.startswith(f'{str(iteration)}_{str(prompt_id)}')
        ]
        self.prompt_lst = [get_prompt(subject_name, prompt_id)] * len(self.image_lst)
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs


class PairwiseImageDatasetCLIP(Dataset):
    def __init__(
            self, subject_name, data_dir_A, data_dir_B, processor,
            iteration, prompt_id,
        ):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        # subject_name, prompt_idx = subject_name.split('-')
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        # data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_files_B = [
            os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B)
            if f.endswith(".png") and f.startswith(f'{str(iteration)}_{str(prompt_id)}')
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject_name, data_dir_A, data_dir_B, feature_extractor, iteration, prompt_id):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        # subject_name, prompt_idx = subject_name.split('-')
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        # data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_files_B = [
            os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B)
            if f.endswith(".png") and f.startswith(f'{str(iteration)}_{str(prompt_id)}')
        ]
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject_name, data_dir_A, data_dir_B, iteration, prompt_id):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        # subject_name, prompt_idx = subject_name.split('-')
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        # data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        # self.image_files_B = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        self.image_files_B = [
            os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B)
            if f.endswith(".png") and f.startswith(f'{str(iteration)}_{str(prompt_id)}')
        ]
        
        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


def clip_text(subject_name, image_dir, prompt_id):
    criterion = 'clip_text'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the text features
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    best_mean_similarity = 0
    mean_similarity_list = []
    for it in EVAL_ITER:
        similarity = []
        dataset = PromptDatasetCLIP(
            subject_name, image_dir, tokenizer, processor, it, prompt_id)
        # dataloader = DataLoader(dataset, batch_size=32)
        for i in range(len(dataset)):
            image_inputs, prompt_inputs = dataset[i]
            if image_inputs is not None and prompt_inputs is not None:
                image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
                # print(prompt_inputs)
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**prompt_inputs)

                sim = cosine_similarity(image_features, text_features)

                #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                #logit_scale = model.logit_scale.exp()
                #sim = torch.matmul(text_features, image_features.t()) * logit_scale
                similarity.append(sim.item())

        if similarity:
            mean_similarity = torch.tensor(similarity).mean().item()
            mean_similarity_list.append(mean_similarity)
            best_mean_similarity = max(best_mean_similarity, mean_similarity)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')
        else:  
            mean_similarity_list.append(0)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {0}({best_mean_similarity})')

    return mean_similarity_list


def clip_image(subject_name, image_dir, prompt_id, dreambooth_dir='dataset/Customization'):
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    best_mean_similarity = 0
    mean_similarity_list = []
    for it in EVAL_ITER:
        similarity = []
        dataset = PairwiseImageDatasetCLIP(
            subject_name, dreambooth_dir, image_dir, processor, it, prompt_id)

        for i in range(len(dataset)):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
            
                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())
                    
        if similarity:
            mean_similarity = torch.tensor(similarity).mean().item()
            best_mean_similarity = max(best_mean_similarity, mean_similarity)
            mean_similarity_list.append(mean_similarity)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')
        else:  
            mean_similarity_list.append(0)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {0}({best_mean_similarity})')

    return mean_similarity_list


def dino(subject_name, image_dir, prompt_id, v2=True, dreambooth_dir='dataset/Customization'):
    if v2:
        criterion = 'dinov2'
    else:
        criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if v2:
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base',)
    else:
        model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

    best_mean_similarity = 0
    mean_similarity_list = []
    for it in EVAL_ITER:
        similarity = []
        dataset = PairwiseImageDatasetDINO(
            subject_name, dreambooth_dir, image_dir, feature_extractor, it, prompt_id)

        for i in range(len(dataset)):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

        mean_similarity = torch.tensor(similarity).mean().item()
        best_mean_similarity = max(best_mean_similarity, mean_similarity)
        mean_similarity_list.append(mean_similarity)
        print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')

    return mean_similarity_list


def lpips_image(subject_name, image_dir, prompt_id, dreambooth_dir='dataset/Customization'):
    criterion = 'lpips_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the LPIPS model (vgg=True uses the VGG-based model from the paper)
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    mean_similarity_list = []
    best_mean_similarity = 0
    for it in EVAL_ITER:
        similarity = []
        dataset = PairwiseImageDatasetLPIPS(
            subject_name, dreambooth_dir, image_dir, it, prompt_id)
        # dataset = SelfPairwiseImageDatasetLPIPS(subject, './data')
        
        for i in range(len(dataset)):
            image_A, image_B = dataset[i]
            if image_A is not None and image_B is not None:
                image_A = image_A.to(device)
                image_B = image_B.to(device)

                # Calculate LPIPS between the two images
                distance = loss_fn(image_A, image_B)

                similarity.append(distance.item())

        mean_similarity = torch.tensor(similarity).mean().item()
        best_mean_similarity = max(best_mean_similarity, mean_similarity)
        mean_similarity_list.append(mean_similarity)
        print(f'epoch: {it}, criterion: LPIPS distance, mean_similarity: {mean_similarity}({best_mean_similarity})')

    return mean_similarity_list        


def siglip(subject_name, image_dir, prompt_id):
    criterion = 'SigLIP'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model = AutoModel.from_pretrained("google/siglip-base-patch16-512").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")

    best_mean_similarity = 0
    mean_similarity_list = []
    for it in EVAL_ITER:
        similarity = []
        dataset = PromptDatasetSigLIP(
            subject_name, image_dir, processor, it, prompt_id)
        # dataloader = DataLoader(dataset, batch_size=32)
        for i in range(len(dataset)):
            inputs = dataset[i]
            if inputs is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(device)
                inputs['input_ids'] = inputs['input_ids'].to(device)

                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits_per_image)[0][0]

                similarity.append(probs.item())

        if similarity:
            mean_similarity = torch.tensor(similarity).mean().item()
            mean_similarity_list.append(mean_similarity)
            best_mean_similarity = max(best_mean_similarity, mean_similarity)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')
        else:  
            mean_similarity_list.append(0)
            print(f'epoch: {it}, criterion: {criterion}, mean_similarity: {0}({best_mean_similarity})')

    return mean_similarity_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--subject", type=str, default="backpack_dog")
    parser.add_argument("--image_dir", type=str, default="output/lora/rank_1")
    args = parser.parse_args()

    # EVAL_ITER = [u * 200 for u in range(1, 31)]
    EVAL_ITER = [u * 200 for u in range(1, 11)]
    NUM_SAMPLES = 8
    SUBJECT = args.subject

    image_dir = args.image_dir
    
    # subject_dirs, subject_names = [], []
    # for name in os.listdir(image_dir):
    #     if os.path.isdir(os.path.join(image_dir, name)):
    #         subject_dirs.append(os.path.join(image_dir, name))
    #         subject_names.append(name)
    
    results_path = os.path.join(image_dir, SUBJECT, 'true_results.json')
    # {'backpack-0':{'DINO':[x, ...], 'CLIP-I':[x, ...], 'CLIP-T':[x, ...], 'LPIPS':[x, ...],}}
    
    results_dict = dict()
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = f.__iter__()
            while True:
                try:
                    result_json = json.loads(next(results))
                    results_dict.update(result_json)
                    
                except StopIteration:
                    print("finish extraction.")
                    break
    
    subject_dir = os.path.join(image_dir, SUBJECT)

    for prompt_id in range(10):
        # if SUBJECT in results_dict:  # TODO
        #     raise RuntimeError("Results exist, please check...")
        
        print(f'evaluating {subject_dir}')
        dino_sim = dino(SUBJECT, subject_dir, prompt_id, v2=False)
        dino_sim_v2 = dino(SUBJECT, subject_dir, prompt_id, v2=True)
        clip_i_sim = clip_image(SUBJECT, subject_dir, prompt_id)
        clip_t_sim = clip_text(SUBJECT, subject_dir, prompt_id)
        lpips_sim = lpips_image(SUBJECT, subject_dir, prompt_id)

        subject_result = {
            'DINO': dino_sim,
            'DINOv2': dino_sim_v2,
            'CLIP-I': clip_i_sim,
            'CLIP-T': clip_t_sim,
            'LPIPS': lpips_sim,
        }
        print(subject_result)
        results_dict.update({f'{SUBJECT}-{prompt_id}': subject_result})

        # with open(results_path,'a') as f:
        #     json_string = json.dumps({f'{SUBJECT}-{prompt_id}': subject_result})
        #     f.write(json_string + "\n")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f)
        # f.write(results_dict)
        # with open('result.json', 'w') as fp:
