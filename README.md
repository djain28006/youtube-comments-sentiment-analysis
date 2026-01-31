# YouTube Comment Sentiment Analysis (Bi-LSTM)

<p align="center">
  <img src="Screenshot 2026-01-31 235230.png" alt="YouTube Emotion & Insight Analysis UI" width="900">
</p>

This repository contains a self-trained YouTube Comment Sentiment Analysis system built using a Bi-Directional LSTM with TensorFlow & Keras.  
The goal of this project is to understand sentiment modeling from first principles and deploy a complete, end-to-end ML application.

---
## Overview

The application follows a simple workflow:

1. User pastes a YouTube video link  
2. Top 20 comments are fetched using the YouTube Data API  
3. The trained model analyzes these comments to infer the overall sentiment of the video discussion

This version of the project intentionally avoids pre-trained transformer models to focus on core NLP concepts.

---

## Model Details

- Architecture: Bi-Directional LSTM (RNN)
- Framework: TensorFlow & Keras
- Text Representation: Embedding Layer
- Training Data: English-language sentiment dataset

The model performs well on practical inputs but shows clear overfitting, especially when trained on limited data. This highlights the challenges of building robust language models from scratch.

Although trained on English data, the model is able to handle Hinglish (mixed Hindi-English) comments to some extent.

---

## Tech Stack

- Python  
- TensorFlow & Keras  
- YouTube Data API  
- FastAPI (Backend API)  
- Streamlit (Frontend UI)

---

## Live Application

🚀 Deployed Streamlit App  
https://youtube-comments-sentiment-analysis-ptochpwhzevotqhfvdntti.streamlit.app/

---

## Project Motivation

This project was built to:
- Learn sequence modeling and embeddings in NLP
- Understand limitations of self-trained models
- Compare traditional RNN approaches with modern transformer-based systems

A transformer-based version of this project is available separately for benchmarking and comparison.

---

## Author

Danish Jain  
GitHub: https://github.com/djain28006
