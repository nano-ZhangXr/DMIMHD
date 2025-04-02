# DMIMHD: Dynamic Modality Interaction for Multi-modal Human Desire Understanding

This repository contains the official implementation of the paper:

**"DMIMHD: Dynamic Modality Interaction for Multi-modal Human Desire Understanding"**  
by **Xiangrui Zhang**  
Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University

[[Paper (PDF)]](./DMIMHD.pdf)

---

## 🧠 Introduction

Understanding human desires, emotions, and sentiments through multi-modal data is a challenging task. Our model **DMIMHD** introduces a unified **Dynamic Capsule Network** that captures interactions between textual and visual features using four specialized interaction units:
- Rectified Identity Unit (RIU)
- Semantic Relation Unit (SRU)
- Contextual Guidance Unit (CGU)
- Cross-modal Enhancement Unit (CMEU)

Each unit uses a dynamic soft selector to conditionally activate based on the input, allowing the model to adaptively fuse information.

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

## 🔧 Installation

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

We use the **MSED** dataset (Multi-modal and Multi-task Sentiment Emotion and Desire), proposed by Jia et al. (2022), which includes:
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

DMIMHD achieves state-of-the-art performance on all tasks compared to 12 strong baselines including BERT+ResNet and Multi-modal Transformers. See our paper for detailed numbers.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2025dmimhd,
  title={DMIMHD: Dynamic Modality Interaction for Multi-modal Human Desire Understanding},
  author={Xiangrui Zhang},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
  note={In press}
}
```

---

## 📬 Contact

For questions or collaborations, feel free to reach out:

**Email:** xiangruizhang00@gmail.com
