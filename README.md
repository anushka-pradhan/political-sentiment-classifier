# 🏛️ Political Sentiment Analysis on Social Media

A comparative study of machine learning and deep learning approaches for analyzing political sentiment across social media platforms.

## Overview

This project addresses critical challenges in political sentiment analysis: lack of contextual understanding, domain generalizability issues, and temporal drift as political discourse evolves. I implemented and evaluated multiple architectures—from traditional machine learning to state-of-the-art transformers—to identify models that balance accuracy, generalizability, and computational efficiency for real-time political discourse monitoring.

## 📊 Dataset

- **Source**: 9,122 political posts scraped from BlueSky (a decentralized social media platform)
- **Collection Date**: October 10th, 2025
- **Keywords**: Elections, legislation, world events, politicians
- **Features**: Post text, timestamps, engagement metrics (likes, replies, reposts), author information
- **Labels**: Three-class sentiment (Positive, Negative, Neutral)
- **Split**: 80-10-10 train-validation-test with stratified sampling

## Data Preprocessing Pipeline

The preprocessing pipeline handles the complexities of political discourse on social media:

- Emoji-to-text conversion
- Hashtag normalization (removing `#` while retaining text)
- URL removal
- Text standardization (lowercase conversion)
- Stopword removal and lemmatization (for applicable models)
- Tokenization and padding for neural network inputs

## Models Implemented

### Traditional Machine Learning
- **Random Forest**: With DistilBERT embeddings for semantic encoding
- **Gradient Boosting**: With TF-IDF vectorization (8,000 features, bigrams)

### Deep Learning
- **CNN-LSTM**: Combines convolutional layers for local pattern extraction with LSTM for sequential dependencies (GloVe embeddings)
- **Bidirectional LSTM (BiLSTM)**: Processes text bidirectionally to capture context from both directions
- **Fine-tuned BERT**: Transformer-based model using `bert-base-uncased`

## Results

Performance on ChatGPT-labeled dataset:

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| BiLSTM (RNN) | 96.5% | 95.0% | **96.0%** |
| Gradient Boosting | 98.1% | 96.7% | 95.8% |
| CNN-LSTM | 97.4% | 89.8% | 90.5% |
| Fine-Tuned BERT | 96.6% | 88.7% | 89.0% |
| Random Forest | 100% | 72.4% | 75.0% |

Performance on hand-labeled subset (~200 samples):

| Model | Test Acc |
|-------|----------|
| BiLSTM (RNN) | **97.0%** |
| Gradient Boosting | 96.5% |
| Fine-Tuned BERT | 86.4% |
| Random Forest | 85.6% |
| CNN-LSTM | 84.3% |

**Note**: Results may vary slightly from those reported in the paper due to model rebuilding and random seed variations. We prioritized reproducibility while acknowledging some inherent variance in neural network training.

## Key Findings

### Best Performing Models
1. **BiLSTM**: Most stable performance across all splits with minimal overfitting (96.0% test accuracy)
2. **Gradient Boosting**: Strong traditional baseline (95.8% test accuracy) with significantly lower computational requirements

### Model Insights
- **BiLSTM's bidirectional architecture** effectively captures contextual information from both preceding and following words, crucial for political sentiment
- **Gradient Boosting** models complex non-linear relationships while maintaining computational efficiency—ideal for real-time monitoring of high-volume events
- **Random Forest** suffered from severe overfitting despite using transformer embeddings
- **CNN-LSTM** showed sensitivity to mixed labeling sources (ChatGPT vs. human labels)

### Computational Trade-offs
The performance gap between Gradient Boosting (95.8%) and BiLSTM (96.0%) is minimal, but computational requirements differ dramatically:
- **Gradient Boosting**: Processes thousands of posts/second on standard hardware
- **Deep Learning Models**: Require GPU acceleration and substantial memory

## Challenges & Limitations

### Data Labeling
- **Neutral-label bias**: ChatGPT labeling introduced systematic skew toward neutral classifications due to model safety mechanisms
- **Label inconsistency**: Mixing ChatGPT-labeled and hand-labeled data created training signal conflicts
- **Ambiguity**: Many posts reasonably fall into multiple sentiment categories

### Model Limitations
- Struggled with posts containing statistics without interpretation
- Difficulty with non-English posts using romanized script
- Limited understanding of poetic devices, metaphors, and sarcasm
- Platform-specific patterns may limit cross-platform generalizability

## Future Work

- **Scale manual labeling** to full dataset to reduce neutral-label bias
- **Cross-platform evaluation** (Twitter, Reddit, Facebook) to test generalizability
- **Temporal drift analysis** to quantify performance degradation over time
- **Fine-grained emotion classification** beyond positive/negative/neutral (anger, hope, fear, enthusiasm, disgust)
- **Investigate CNN-LSTM sensitivity** to mixed labeling sources

## Broader Impact

This work has implications for:
- **Political analysts & campaigns**: Real-time public opinion monitoring during evolving events
- **Researchers**: Evidence-based model selection for political discourse analysis
- **Practitioners**: Framework for balancing accuracy vs. computational efficiency
- **NLP community**: Cautionary insights on LLM-based annotation biases
