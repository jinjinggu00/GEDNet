# Adverse Multi-Weather Image Restoration for Boosting Downstream Object Detection



> [**Adverse Multi-Weather Image Restoration for Boosting Downstream Object Detection**]<br>
> [Jinjing Gu*], [Chenggang Yang], [Yuanyuan Pu], [Zhengpeng Zhao]
> Adverse weather conditions severely degrade image quality, posing a critical challenge to robust object detection in real-world outdoor vision applications. Current paradigms commonly employ monolithic designs that tightly integrate image restoration with weather understanding, which restricts both flexibility and scalability. To overcome this, we propose a novel modular framework named Gating-Enhancement-Detection Network(GEDNet), which decouples weather identification from image restoration. First, a Weather Classification Gating(WCG) module that synergizes a hierarchical feature backbone with a frequency-domain attention mechanism is meticulously designed to achieve robust global context modeling for accurate weather identification. Based on the WCG output, an interchangeable mixture-of-experts-based restoration module is dynamically dispatched to perform targeted image restoration. Finally, the restored image is fed into a downstream detector for robust object localization. Comprehensive evaluations across multiple benchmarks demonstrate that GEDNet consistently outperforms existing approaches in both visual quality and quantitative performance. Notably, on the challenging ACDC dataset, our framework improves the mean Average Precision by an absolute margin of 0.86%, demonstrating the significant practical efficacy of the GEDNet paradigm.

## Requirements
- Python 3.10+
```pip install -r requirements.txt```
