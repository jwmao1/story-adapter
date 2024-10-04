<p align="center">
  <img src="https://github.com/jwmao1/story-adapter/docs\docs\logo.png" height=150>
</p>



# Story-Adapter: A Training-free Iterative Framework for Long Story Visualization
<span>
<a href="https://arxiv.org/abs/2406.12587"><img src="https://img.shields.io/badge/arXiv-2406.12587-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  
<a href="https://arxiv.org/abs/2406.12587"><img src="https://img.shields.io/badge/project-StoryAdapter-purple.svg" height=22.5></a>
</span>

Code for the paper [Story-Adapter: A Training-free Iterative Framework for Long Story Visualization](https://arxiv.org/abs/2406.12587)

Note: This code base is still not complete. Since stable diffusion cannot provide the corresponding style of initialized reference images for subsequent iteration stably, story visualization style of Story-Adapter may be affected.

### About this repo:

The repository contains the official implementation of "Story-Adapter".

## Introduction 🦖

> Story visualization, the task of generating coherent images based on a narrative, has seen significant advancements with the emergence of text-to-image models, particularly diffusion models. However, maintaining semantic consistency, generating high-quality fine-grained interactions, and ensuring computational feasibility remain challenging, especially in long story visualization (_i.e._, up to 100 frames). In this work, we propose a training-free and computationally efficient framework, termed **Story-Adapter**, to enhance the generative capability of long stories. Specifically, we propose an _iterative_ paradigm to refine each generated image, leveraging both the text prompt and all generated images from the previous iteration. Central to our framework is a training-free global reference cross-attention module, which aggregates all generated images from the previous iteration to preserve semantic consistency across the entire story, while minimizing computational costs with global embeddings. This iterative process progressively optimizes image generation by repeatedly incorporating text constraints, resulting in more precise and fine-grained interactions. Extensive experiments validate the superiority of Story-Adapter in improving both semantic consistency and generative capability for fine-grained interactions, particularly in long story scenarios.

<br>

<img src="https://github.com/jwmao1/story-adapter/docs/teaser-github.jpg" width="800"/>


## News 🚀
* **2024.10.21**: Code released.
* **2024.10.03**: [Paper](https://arxiv.org/abs/2406.12587) is released on ArXiv.

## Framework 🤖 

> Story-Adapter framework. Illustration of the proposed iterative paradigm, which consists of initialization, iterations in Story-Adapter, and implementation of Global Reference Cross-Attention (GRCA).
Story-Adapter first visualizes each image only based on the text prompt of the story and uses all results as reference images for the future round. 
In the iterative paradigm, Story-Adapter inserts GRCA into SD. For the ith iteration of each image visualization, GRCA will aggregate the information flow of all reference images during the denoising process through cross-attention.
All results from this iteration will be used as a reference image to guide the dynamic update of the story visualization in the next iteration.

<br>

<img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\framework.jpg" width="1080"/>


## Quick Start 🔧

### Installation
The project is built with Python 3.10.14, PyTorch 2.2.2. CUDA 12.1, cuDNN 8.9.02
For installing, follow these instructions:
~~~
# git clone this repository
git clone https://github.com/Talented-Q/Restorer.git
cd StoryAdapter

# create new anaconda env
conda create -n StoryAdapter python=3.10
conda activate StoryAdapter 

# install packages
pip install -r requirements.txt
~~~

### Download the checkpoint
- downloading [RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0/tree/main) put it into "./RealVisXL_V4.0"
- downloading [clip_image_encoder](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/image_encoder) put it into "./IP-Adapter/sdxl_models/image_encoder"
- downloading [ip-adapter_sdxl](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin?download=true) put it into "./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

### Running Demo

~~~
python run.py --base_model_path your_path/RealVisXL_V4.0 --image_encoder_path your_path/IP-Adapter/sdxl_models/image_encoder --ip_ckpt your_path//IP-Adapter/sdxl_models/ip-adapter_sdxl.bin 
~~~

### Customized Running

~~~
python run.py --base_model_path your_path/RealVisXL_V4.0 --image_encoder_path your_path/IP-Adapter/sdxl_models/image_encoder --ip_ckpt your_path//IP-Adapter/sdxl_models/ip-adapter_sdxl.bin 
--story [your story] 
~~~

## Performance 🎨

### Regular-length Story Visualization 
- downloading the [StorySalon](https://huggingface.co/datasets/haoningwu/StorySalon/resolve/main/testset.zip?download=true) test set."

| GIF1 | GIF2  | GIF3  |
| --- | --- | --- |
| <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_005169.gif" alt="GIF 1" width="224"/>  | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_007016.gif" alt="GIF 2" width="224"/> | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_007137.gif" alt="GIF 3" width="224"/>  |

| GIF4 | GIF5  | GIF6  |
| --- | --- | --- |
| <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_013804.gif" alt="GIF 4" width="224"/>  | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_015770.gif" alt="GIF 5" width="224"/> | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_000026.gif" alt="GIF 6" width="224"/>  |

| GIF7 | GIF8  | GIF9  |
| --- | --- | --- |
| <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_012060.gif" alt="GIF 7" width="224"/>  | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_008614.gif" alt="GIF 8" width="224"/> | <img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\our_008710.gif" alt="GIF 9" width="224"/>  |


### Long Story Visualization 

<br>

<img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\comic1.png" width="1080"/>

<br>
<br>
<img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\comic7.png" width="1080"/>

<br>
<br>
<img src="C:\Users\86187\Desktop\story-adapter\storyadapter\docs\comic3.png" width="1080"/>

## Acknowledgement 🍻

Deeply appreciate these wonderful open source projects: [stablediffusion](https://github.com/Stability-AI/StableDiffusion), [clip](https://github.com/openai/CLIP), [ip-adapter](https://github.com/tencent-ailab/IP-Adapter), [storygen](https://github.com/haoningwu3639/StoryGen), [storydiffusion](https://github.com/HVision-NKU/StoryDiffusion), [timm](https://github.com/huggingface/pytorch-image-models).

## Citation 🔖

If you find this repository useful, please consider giving a star ⭐ and citation 🙈:

```
@inproceedings{huang2024segment,
  title={Segment and caption anything},
  author={Huang, Xiaoke and Wang, Jianfeng and Tang, Yansong and Zhang, Zheng and Hu, Han and Lu, Jiwen and Wang, Lijuan and Liu, Zicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13405--13417},
  year={2024}
}
```





