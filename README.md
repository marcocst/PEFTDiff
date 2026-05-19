## Overview

Code for the paper [[arXiv]](https://arxiv.org/abs/2501.08727)

**Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models (ICCV-25)**

by Zerui Tao, Yuhta Takida, Naoki Murata, Qibin Zhao, and Yuki Mitsufuji

The code for subject-driven generation and controllable generation experiments are stored in `./subject-driven-generation` and `./control` respectively.

## Subject-driven generation

- Dataset: The dataset can be downloaded from [here](https://github.com/phymhan/SODA-Diffusion/tree/f15ded29db3a545dc7203d1c16fb90cf873cc227/dataset/Customization). Please store the dataset in `./subject-driven-generation/dataset/Customization`.
- Experiment: `cd ./subject-driven-generation && bash run_tlora.sh`.

## Controllable generation

We follow the guideline of [OFT](https://github.com/zqiu24/oft) for data processing, training, and evaluation. After downloading and processing the datasets, please run the following example scripts for exepriments:

- `bash tt_control_build.sh`: Build the model.
- `bash tt_control_train.sh`: Train the model with images and controlling signals.
- `bash tt_control_test.sh`: Generate test images given test controlling signals.
- `python eval_{canny,landmark,segm}.sh`: Evaluations for corresponding tasks.

## Acknowledgements

Our codebase is implemented based on the following projects. Thanks for their contributions.

- DCO: https://github.com/kyungmnlee/dco
- SODA: https://github.com/phymhan/SODA-Diffusion
- OFT: https://github.com/zqiu24/oft

## Citation

```
@inproceedings{tao2025transformed,
  title={Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models},
  author={Tao, Zerui and Takida, Yuhta and Murata, Naoki and Zhao, Qibin and Mitsufuji, Yuki},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month={October},
  year={2025}
}
```