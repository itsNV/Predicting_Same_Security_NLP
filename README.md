# Same Security Classification using NLP

## 📌 Project Overview
This project focuses on **Same Security Classification**, where the goal is to determine whether two financial security descriptions refer to the **same underlying security**.  
The problem is framed as a **binary classification task** using Natural Language Processing (NLP) techniques and machine learning / deep learning models.

---

## 🧾 Dataset Description
The dataset contains the following features:

- **description_x**: Text description of the first security  
- **description_y**: Text description of the second security  
- **token_x**: Tokenized version of `description_x`  
- **token_y**: Tokenized version of `description_y`  
- **same_security (target)**:  
  - `1` → Both descriptions refer to the same security  
  - `0` → Different securities  

For model training (LSTM/RNN), **only `description_x` and `description_y` are used as input features**, while token-based columns are used during preprocessing and experimentation.

---

## Text Preprocessing
The following NLP preprocessing steps were applied:

- Lowercasing
- Removal of special characters and punctuation
- Stopword removal
- Tokenization
- Lemmatization
- Text sequence padding (for LSTM models)

---

## Feature Engineering & Vectorization
Multiple vectorization techniques were explored and evaluated:

1. **Bag of Words (BoW)**
2. **TF-IDF (Term Frequency–Inverse Document Frequency)**
3. **Word2Vec (Dense Word Embeddings)**

Each vectorization method was trained and evaluated independently to compare performance.

---

## 🧠 Models Used

### 🔹 Deep Learning
- **LSTM (RNN-based model)**  
  - Used combined text from `description_x` and `description_y`
  - Captures sequential and contextual information in text

### 🔹 Machine Learning
- **K-Nearest Neighbors (KNN) Classifier**
  - Used with BoW, TF-IDF, and Word2Vec vectors
  - Distance-based similarity learning for classification

---

## 📊 Model Evaluation
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Performance comparison was done across different vectorization techniques to identify the most effective approach.

---

## 🗂️ Project Structure

├── data/

│ └── train.csv

│ └── train.csv

├── notebooks/

│ └── Same_Security.ipynb

├── README.md



---

## 🚀 Key Learnings
- Importance of text preprocessing in NLP tasks
- Performance trade-offs between sparse (BoW, TF-IDF) and dense (Word2Vec) representations
- Effectiveness of LSTM models in capturing semantic similarity
- Distance-based classifiers like KNN can perform competitively with proper vectorization

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / SpaCy
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## Future Improvements
- Experiment with **Siamese LSTM / Bi-LSTM**
- Use **Transformer-based models (BERT, SBERT)**

