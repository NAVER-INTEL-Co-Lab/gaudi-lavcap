# Speech LLM Framework for Automatic Speech Recognition
This repository provides a framework for a Large Language Model (LLM) that supports spoken data inputs, i.e., recognizing speech. This framework adapts the LLaMA-2 model to process spoken inputs by leveraging Low Rank Adaptation (LoRA) on Gaudi-2 nodes.
## Prerequisites
### 1. Download pre-trained LLaMA-2
We use the LLaMA-2 model as the backbone foundation model, specifically the `llama-2-7b-chat-hf` variant, which can be downloaded from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
### 2. Download TED-LIUM 3 dataset
The `TED-LIUM 3` dataset is used for training and inference. You can download the dataset from [here](https://www.openslr.org/51/).
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
The configuration for training and inference is specified in the ```./configs/asr.yaml``` file.
## How to train
### Single-node training
To train the model on a single node, run:
```
PT_HPU_LAZY_MODE=0 python train.py --cfg-path configs/asr.yaml
```
### Multi-node training
To train on multiple nodes, run:
```
PT_HPU_LAZY_MODE=0 python gaudi_spawn.py --world_size 8 --use_mpi train.py --cfg-path configs/asr.yaml
```
## How to test
To test the mode using a trained model checkpoint (the result of the above training process), replace /path/to/ckpt with the actual path to your checkpoint, and run
```
PT_HPU_LAZY_MODE=0 python train.py --cfg-path configs/asr.yaml --options run.do_train=False run.do_test=True model.ckpt=/path/to/ckpt
```
## Acknowledgement
The codes are based on the [official repository](https://github.com/bytedance/SALMONN/tree/main) of [SALMONN](https://openreview.net/pdf?id=14rn7HpKVk).