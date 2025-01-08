# LeetCode-Topics-Classification-Using-BERT-Based-NLP-Models

## Overview
This project aims to classify LeetCode problems by their topics and difficulty levels using advanced Natural Language Processing (NLP) techniques, specifically leveraging the pre-trained BERT model. The dataset includes problem descriptions, topics, and difficulty metadata sourced from LeetCode.

## Objectives
- Classify LeetCode problems into relevant topics (e.g., Array, Graph, Dynamic Programming).
- Predict the difficulty level of a problem (Easy, Medium, Hard) based on its description.
- Utilize a fine-tuned BERT model for text encoding and classification.

## Dataset
The dataset consists of:
- **Problem Descriptions:** Detailed text descriptions of coding problems.
- **Topics:** Tags assigned to each problem (e.g., Array, Hash Table).
- **Difficulty Levels:** Categorical difficulty labels (Easy, Medium, Hard).

Data was fetched using GraphQL APIs from LeetCode and pre-processed for training.

## Technologies Used
- **Python**: Primary programming language.
- **PyTorch**: Deep learning library for building and training neural networks.
- **Transformers (HuggingFace)**: For utilizing pre-trained BERT models.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **GraphQL API**: Fetching LeetCode problem data.

## Model Architecture
- **BERT (Bidirectional Encoder Representations from Transformers):** Used to encode problem descriptions.
- **Multi-label Classification Head:** Predicts problem topics.
- **Difficulty Embedding:** Encodes problem difficulty levels as embeddings for enhanced prediction.

## Project Structure
```
.
├── data/
│   ├── leetcode_problems.json
│   ├── filtered_leetcode_problems.json
│
├── models/
│   ├── bert_multilabel_model.py
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── training.ipynb
│   ├── evaluation.ipynb
│
├── results/
│   ├── training_losses.png
│   ├── validation_losses.png
│
├── main.py
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/leetcode-nlp-classifier.git
   cd leetcode-nlp-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure access to the HuggingFace token for model authentication.

## Usage
1. **Data Collection:** Fetch LeetCode data using GraphQL API.
2. **Preprocessing:** Preprocess the data using the provided scripts.
3. **Training:** Fine-tune the BERT model on the dataset.
4. **Evaluation:** Evaluate the model on validation data.
5. **Prediction:** Run predictions on unseen data.

## Results
- Training and validation loss plots.
- Accuracy and F1 scores for topic classification and difficulty prediction.

## Future Improvements
- Fine-tune on larger datasets.
- Add support for more granular difficulty levels.
- Experiment with other transformer-based architectures.

## Contact
For questions or collaboration:
- Email: zacheudavi@gmail.com
- GitHub: [davizacheu](https://github.com/davizacheu)


