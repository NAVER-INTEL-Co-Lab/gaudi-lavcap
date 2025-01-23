# LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport
This repository provides the pytorch source code for ICASSP 2025 paper [LAVCap](https://www.arxiv.org/abs/2501.09291), optimized for the use with Intel Gaudi-v2 accelerator.

## Prerequisites
### 1. Download pre-trained LLaMA-2
LAVCap leverages the `llama-2-7b-chat-hf` variant of the LLaMA-2 model as its foundational backbone. You can download the model from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

### 2. Download AudioCaps dataset
The `AudioCaps` dataset is used for training and evaluation. You can download the dataset from [here](https://github.com/cdjkim/audiocaps).

### 3. Create a Docker container
Run the following command to create a Docker container.
```
bash ./sllm_docker.sh 
```
For more details, refer to the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html)

### 4. Install dependencies
Install the necessary dependencies by running:
```
pip install -r ./requirements.txt
``` 

### 5. Set the configuration
The configuration for training is specified in the ```./configs/lavcap.yaml``` file. Modify this file as needed before proceeding.
## How to train
### Single-node training
To train the model on a single node, run:
```
PT_HPU_LAZY_MODE=0 python train.py --cfg-path configs/lavcap.yaml
```
### Multi-node training
To train on multiple nodes, run:
```
PT_HPU_LAZY_MODE=0 python gaudi_spawn.py --world_size 8 --use_mpi train.py --cfg-path configs/lavcap.yaml
```
## How to test
To test the mode using a trained model checkpoint (the result of the above training process), replace /path/to/ckpt with the actual path to your checkpoint, and run
```
PT_HPU_LAZY_MODE=0 python train.py --cfg-path configs/lavcap.yaml --options run.do_eval=True model.resume_from=/path/to/ckpt
```
## Acknowledgement
- This project was developed with support from the NAVER-Intel Co-Lab.
