# Seismic Event Classification using CNN

A 1D Convolutional Neural Network for classifying earthquake signals
vs background seismic noise, trained on the STEAD global seismic dataset.
Achieved **98% accuracy** on 400 held-out test waveforms.

---

## Results

| Metric               | Value        |
| -------------------- | ------------ |
| Test Accuracy        | 98%          |
| Earthquake Precision | 0.99         |
| Earthquake Recall    | 0.97         |
| Noise Precision      | 0.98         |
| Noise Recall         | 0.99         |
| F1 Score             | 0.98         |
| Wrong Predictions    | 7 out of 400 |

### Confusion Matrix

```text
[[198   2]
 [  5 195]]
```

---

## Model Architecture

```text
Input (6000, 1)
        ↓
Conv1D (32 filters, kernel=5) + BatchNorm + MaxPooling
        ↓
Conv1D (64 filters, kernel=3) + BatchNorm + MaxPooling
        ↓
Conv1D (128 filters, kernel=3) + MaxPooling
        ↓
Flatten
        ↓
Dense (64, ReLU) + Dropout (0.5)
        ↓
Dense (1, Sigmoid) → Probability Score
```

* Loss Function: Binary Crossentropy
* Optimizer: Adam with ReduceLROnPlateau scheduler
* Training: 20 epochs, batch size 64

---

## Dataset

This project uses the **STEAD (STanford EArthquake Dataset)** —
a global dataset of seismic signals for AI research.

Download the following files and place them inside the project folder:

| File        | Label                | Size     | Link                      |
| ----------- | -------------------- | -------- | ------------------------- |
| chunk1.hdf5 | Noise Waveforms      | ~14.6 GB | https://rebrand.ly/chunk1 |
| chunk2.hdf5 | Earthquake Waveforms | ~13.7 GB | https://rebrand.ly/chunk2 |

Expected folder structure after download:

```text
seismic-cnn/
├── chunk1/
│   ├── chunk1.hdf5
│   └── chunk1.csv
├── chunk2/
│   ├── chunk2.hdf5
│   └── chunk2.csv
├── seismic_cnn.ipynb
├── README.md
└── .gitignore
```

Then update the file paths in Cell 2 of the notebook to match
your local setup.

---

## Also Tested on Real K-NET Data

In addition to STEAD, this model was tested on real K-NET
accelerometer recordings from the **2011 Tohoku M9.0 Earthquake**
(NIED K-NET network, 100 stations, Aichi prefecture).

**Task:** Strong Shaking vs Coda Wave Classification

**Result:** **85% accuracy** on 40 test samples across 100 stations.

---

## Setup

### Requirements

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib h5py jupyter
```

### Run

```bash
jupyter notebook
```

Open `seismic_cnn.ipynb` and run Cells 1–10 in order.

> **Note:** Cell 7 (training) takes approximately 30–60 minutes on CPU.
> Enable GPU acceleration for faster training.

---

## What the Notebook Covers

| Cell    | Description                                |
| ------- | ------------------------------------------ |
| Cell 1  | Import libraries                           |
| Cell 2  | Load STEAD waveforms from HDF5             |
| Cell 3  | Visualize earthquake vs noise waveforms    |
| Cell 4  | Train/test split                           |
| Cell 5  | Build CNN architecture                     |
| Cell 6  | Compile model                              |
| Cell 7  | Train model and monitor loss               |
| Cell 8  | Plot loss and accuracy curves              |
| Cell 9  | Classification report and confusion matrix |
| Cell 10 | Single waveform prediction                 |
---


## Training History of the CNN Model
<img width="1342" height="399" alt="image" src="https://github.com/user-attachments/assets/2dd44197-a5b7-4a73-bf97-b4a82983b012" />


---

## Why Some Predictions Are Wrong

The 7 incorrect predictions fall into two categories:

* **False Alarms (2)** — noise recordings containing industrial machinery
  patterns that resemble seismic signals (quarry blasts, generators).
* **Missed Earthquakes (5)** — weak or distant events where the
  seismic signal is nearly buried in background noise.

These are genuinely difficult cases. The ambiguity exists in the
underlying physics of the signal, not merely in the model itself.

---

## Author

**Saatvik Sawarn**
B.E. Computer Science and Engineering,
Chandigarh University (Batch 2024–2028),
Intern at IIT Roorkee, ICED.
