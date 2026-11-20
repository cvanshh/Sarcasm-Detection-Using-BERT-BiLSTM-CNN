# Sarcasm Detection Using BERT + BiLSTM + CNN

A deep learning project that detects sarcasm in text comments using a hybrid neural network architecture combining **BERT, BiLSTM, and CNN**.

The project uses PyTorch for model development and training, with GPU acceleration and mixed-precision training to improve computational efficiency.

## Project Overview

Sarcasm is difficult for machines to detect because the literal meaning of a sentence can differ from its intended meaning.

This project aims to classify text comments into two categories:

- **0 – Not Sarcastic**
- **1 – Sarcastic**

The model uses contextual representations from pretrained BERT, sequential learning through BiLSTM, and feature extraction through CNN.

## Model Architecture

The model follows this architecture:

`Input Text → BERT → Dense Layer → BiLSTM → CNN → Mean Pooling → Dense Layer → Output`

### Components

- **BERT:** Extracts contextual representations from text.
- **BiLSTM:** Captures sequential and contextual information from both directions.
- **CNN:** Extracts important local patterns and features.
- **Dense Layers:** Learn higher-level representations for classification.
- **Output Layer:** Produces a sarcasm prediction.

## Technologies Used

- Python
- PyTorch
- Transformers (Hugging Face)
- BERT (bert-base-uncased)
- BiLSTM
- CNN
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

## Dataset

The project uses `train-balanced-sarcasm.csv`.

Required columns:

| Column | Description |
|---|---|
| label | Sarcasm classification label (0 or 1) |
| comment | Text comment |

The notebook uses the first 10,000 rows by default. This can be changed to use the complete dataset.

## Project Workflow

1. Load the sarcasm detection dataset.
2. Clean and preprocess text comments.
3. Tokenize comments using the BERT tokenizer.
4. Split the dataset into training and testing sets.
5. Build the BERT + BiLSTM + CNN architecture.
6. Train the model using PyTorch.
7. Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
8. Predict sarcasm in new text comments.
9. Save the trained model and tokenizer.

## GPU Acceleration

The notebook supports CUDA-enabled GPU training.

Training optimizations include:

- Automatic GPU detection.
- Mixed-precision training.
- Gradient accumulation.
- Configurable batch size.
- Live training progress using tqdm.

The training configuration is designed to reduce GPU memory usage on GPUs with limited VRAM.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/sarcasm-detection-bert.git
```

Navigate to the project directory:

```bash
cd sarcasm-detection-bert
```

Install the required packages:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

For NVIDIA GPU acceleration, install the appropriate CUDA-enabled PyTorch build from the official PyTorch installation guide.

## Dataset Setup

Place `train-balanced-sarcasm.csv` in the project directory.

The expected structure is:

```text
sarcasm-detection-bert/
│
├── Sarcasm_Detection_GPU_Progress.ipynb
├── train-balanced-sarcasm.csv
└── README.md
```

## Running the Project

Open the notebook in Jupyter Notebook or VS Code.

Run the notebook cells sequentially to preprocess the dataset, train the model, and evaluate its performance.

## Making Predictions

After training, use the prediction function to classify new comments:

```python
text = "Oh great, another meeting. Just what I needed."

prediction, probability = predict_sarcasm(text)

print("Prediction:", prediction)
print("Sarcasm Probability:", probability)
```

Example output:

```text
Prediction: Sarcastic
Sarcasm Probability: 0.87
```

*Example output is illustrative, not a measured result.*

## Model Saving

After training, the notebook saves:

- `sarcasm_bert_pytorch.pth` – trained model weights and configuration.
- `sarcasm_bert_tokenizer/` – saved tokenizer files.

These files can be used to reload the trained model without training it again.

## Future Improvements

- Train on the complete dataset.
- Improve sarcasm detection performance.
- Experiment with different classification thresholds.
- Perform hyperparameter tuning.
- Deploy the model using Streamlit or FastAPI.
- Build a web-based sarcasm detection interface.

## Author

**Shivansh Jha**

B.Tech – Computer Science and Engineering (Data Science)

[LinkedIn](https://www.linkedin.com/in/shivanshjha2509/)

---

If you find this project interesting, consider giving the repository a star!
