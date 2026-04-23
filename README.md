# Adverse Multi-Weather Image Restoration for Boosting Downstream Object Detection



> [**Adverse Multi-Weather Image Restoration for Boosting Downstream Object Detection**]<br>
> [Jinjing Gu*], [Chenggang Yang*], [Yuanyuan Pu*], [Zhengpeng Zhao]
> 
> Adverse weather conditions severely degrade image quality, posing a critical challenge to robust object detection in real-world outdoor vision applications. Current paradigms commonly employ monolithic designs that tightly integrate image restoration with weather understanding, which restricts both flexibility and scalability. To overcome this, we propose a novel modular framework named Gating-Enhancement-Detection Network(GEDNet), which decouples weather identification from image restoration. First, a Weather Classification Gating(WCG) module that synergizes a hierarchical feature backbone with a frequency-domain attention mechanism is meticulously designed to achieve robust global context modeling for accurate weather identification. Based on the WCG output, an interchangeable mixture-of-experts-based restoration module is dynamically dispatched to perform targeted image restoration. Finally, the restored image is fed into a downstream detector for robust object localization. Comprehensive evaluations across multiple benchmarks demonstrate that GEDNet consistently outperforms existing approaches in both visual quality and quantitative performance. Notably, on the challenging ACDC dataset, our framework improves the mean Average Precision by an absolute margin of 0.86%, demonstrating the significant practical efficacy of the GEDNet paradigm.

## Requirements
The code is built with Python 3.10+ and PyTorch. To install the required dependencies, run:
```bash
conda create -n gednet python=3.10
conda activate gednet
pip install -r requirements.txt
```

## Datasets
The experiments are conducted on both classification datasets (for training the WCG module) and detection datasets (for downstream evaluation).\
**WCG Classification Dataset:** A large-scale adverse multi-weather dataset comprising 65,755 images (Clear, Rain, Snow, Fog).\
**Downstream Detection Datasets:** ACDC, DAWN, RTTS, Cityscapes-Rain/Fog, and VOC-Snow.\
Please download the datasets from their official sources and organize them in the``` data/``` directory. Update the``` csv_file``` and ```img_dir paths``` in the configuration file accordingly.

## Training the WCG
We employ a differential learning rate strategy to fine-tune the pre-trained ConvNeXt backbone while training the randomly initialized FDConv module effectively. To start training the WCG module, run:
```
python train.py --config configs/wcg_config.yaml
```

## MoER: Expert Library
One of the core advantages of GEDNet is its modularity. We do not reinvent the wheel for image restoration; instead, we integrate top-tier, independently developed restoration models into our Mixture-of-Experts Restoration (MoER) library.
During our pipeline execution, these experts are kept strictly frozen. You can find the original implementations and pre-trained weights of the selected experts below:
Deraining Expert: https://github.com/Ephemeral182/UDR-S2Former_deraining
Dehazing Expert:
Desnowing Expert:
