# From Data Deluge to Data Curation: A Filtering-WoRA Paradigm for Efficient Text-based Person Search 🕵️‍♂️

[![ArXiv](https://img.shields.io/badge/ArXiv-2404.10292-blue)](https://arxiv.org/abs/2404.10292)

---

## 💡 Introduction

The **Filtering-WoRA** paradigm focuses on efficient **text-based person search** by addressing the challenges of **data deluge** and **data curation**. This innovative approach streamlines the search process, enhancing accuracy and scalability in large-scale applications.

**Authors:**
- Jintao Sun
- Hao Fei
- Gangyi Ding
- Zhedong Zheng

Corresponding authors: Zhedong Zheng

You can read the full paper [here](https://arxiv.org/abs/2404.10292).

---

## 🚀 Getting Started

Start by setting up the **Filtering-WoRA** repository on your local machine. Here's how:

### Prerequisites
Before installing, ensure you have the following dependencies:
- Python 3.x
- PyTorch
- Other dependencies listed in `requirements.txt`

---

## 🛠️ Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/username/Filtering-WoRA.git
cd Filtering-WoRA
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Datasets Prepare

Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) , the PA-100K dataset from [here](https://github.com/xh-liu/HydraPlus-Net), the RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset), and ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN). Download the processed json files of the aboves four datasets from [here](https://pan.baidu.com/s/1oAkenOKaVEYWpNh2hznkGA) [b2l8]

Download pre-trained models for parameter initialization:

image encoder: [swin-transformer-base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)

text encoder: [bert-base](https://huggingface.co/bert-base-uncased/tree/main)

Organize `data` folder as follows:

```
|-- data/
|    |-- bert-base-uncased/
|    |-- finetune/
|        |-- gene_attrs/
|            |-- g_4x_attrs.json
|            |-- g_c_g_a_0_attrs.json
|            |-- ...
|        |-- cuhk_train.json
|        |-- ...
|        |-- icfg_train.json
|        |-- ...
|        |-- rstp_train.json
|        |-- ...
|        |-- PA100K_train.json
|        |-- ...
|    |-- swin_base_patch4_window7_224_22k.pth
```

And organize those datasets in `images` folder as follows:

```
|-- images/
|    |-- <CUHK-PEDES>/
|        |-- imgs/
|            |-- cam_a/
|            |-- cam_b/
|            |-- ...
|            |-- train_query/
|            |-- gene_crop/
|                |-- 4x/
|                |-- c_g_a/
|                |-- ...
|                |-- i_g_a_43/
|
|    |-- <ICFG-PEDES>/
|        |-- test/
|        |-- train/
|
|    |-- <pa100k>/
|        |-- release_data/
|
|    |-- <RSTPReid>/
```

## 📚 Citation

If you use **Filtering-WoRA** in your research, please cite the following BibTeX entry:

```bibtex
@misc{sun2025datadelugedatacuration,
      title={From Data Deluge to Data Curation: A Filtering-WoRA Paradigm for Efficient Text-based Person Search}, 
      author={Jintao Sun and Hao Fei and Zhedong Zheng and Gangyi Ding},
      year={2025},
      eprint={2404.10292},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.10292}, 
}
