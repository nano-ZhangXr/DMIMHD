# 🧩 DMIMHD: Dynamic Modality Interaction for Multi-modal Human Desire Understanding

[![Conference](https://img.shields.io/badge/Conference-IJCNN%202025-blue)](https://ieeexplore.ieee.org/document/11227376)
[![Publisher](https://img.shields.io/badge/Publisher-IEEE-lightgrey)](https://www.ieee.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-brightgreen)](https://doi.org/10.1109/IJCNN64981.2025.11227376)

This repository contains the **official implementation** of the paper:

> **"DMIMHD: Dynamic Modality Interaction for Multi-modal Human Desire Understanding"**  
> **Xiangrui Zhang**  
> *Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University*  
> Published in the *2025 International Joint Conference on Neural Networks (IJCNN)*, Rome, Italy.  
> [DOI: 10.1109/IJCNN64981.2025.11227376](https://doi.org/10.1109/IJCNN64981.2025.11227376)

📄 [[Read on IEEE Xplore]](https://doi.org/10.1109/IJCNN64981.2025.11227376)

---

## 🧠 Introduction

Understanding **human desires** is essential to comprehending human **behavior, sentiment, and emotion**, as desire is one of the most fundamental human instincts.  
Recent advances in **multi-modal learning** have enabled deeper analysis of desires through both **textual and visual** modalities.  
However, existing models often struggle to fully capture **dynamic cross-modal interactions**.

To address this, **DMIMHD** introduces a **unified dynamic modality interaction network** with four specialized interaction units:

- 🔸 **Rectified Identity Unit (RIU)** — preserves unimodal identity features  
- 🔸 **Semantic Relation Unit (SRU)** — learns semantic correspondences between modalities  
- 🔸 **Contextual Guidance Unit (CGU)** — enhances global context alignment  
- 🔸 **Cross-modal Enhancement Unit (CMEU)** — integrates high-level mutual reinforcement

Each unit is equipped with a **soft selector**, allowing **adaptive fusion** of modalities per input sample — effectively learning diverse interaction patterns dynamically.

---

## 🗂️ Project Structure

```
DMIMHD/
├── main.py                 # Main training pipeline
├── Config/                 # Hydra configuration files
├── DynamicModality/        # Custom feature encoders and interaction modules
├── dataset/                # Input CSVs and image folders
├── requirements.txt        # Python dependencies
├── DMIMHD.pdf              # Paper
└── README.md               # This file
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nano-ZhangXr/DMIMHD.git
cd DMIMHD
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
python main.py
```

Make sure your dataset is placed in the `./dataset/` directory as follows:

```
dataset/
├── train/
│   ├── train.csv
│   └── images/
├── dev/
│   ├── dev.csv
│   └── images/
└── test/
    ├── test.csv
    └── images/
```

You can modify training settings in `Config/basic_cfg.yaml`.

---

## 📊 Dataset

We employ the MSED dataset (Multi-modal and Multi-task Sentiment Emotion and Desire) proposed by Jia et al. (2022):
- 6,127 training samples
- 1,021 validation samples
- 2,024 test samples

Each sample is annotated with:
- Desire (7 categories)
- Sentiment (positive, neutral, negative)
- Emotion (6 types)

📎 [Official MSED GitHub Repo](https://github.com/MSEDdataset/MSED)  
📄 Citation:
```bibtex
@inproceedings{jia2022beyond,
  title={Beyond Emotion: A Multi-Modal Dataset for Human Desire Understanding},
  author={Jia, Anqi and He, Yihong and Zhang, Yixuan and Uprety, Sijan and Song, Dawei and Lioma, Christina},
  booktitle={NAACL},
  year={2022}
}
```

---

## 📈 Results

DMIMHD achieves state-of-the-art performance across all tasks compared to 12 competitive baselines (including BERT+ResNet and multi-modal Transformer variants).
Refer to our paper for full quantitative results and ablation studies.

---

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@INPROCEEDINGS{11227376,
  author={Zhang, Xiangrui},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)}, 
  title={DMIMHD: Dynamic Modality Interaction for Multi-Modal Human Desire Understanding}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Adaptation models;Visualization;Limiting;Computational modeling;Neural networks;Behavioral sciences;Human desire understanding;Multimedia;Modality interaction;Dynamic selecting},
  doi={10.1109/IJCNN64981.2025.11227376}}
```

---

## 📬 Contact
For questions, collaborations, or implementation details, feel free to reach out:

👤 Author: Xiangrui Zhang
📧 Email: xiangruizhang00@gmail.com
