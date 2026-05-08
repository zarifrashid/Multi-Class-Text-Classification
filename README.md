# News Classification Using TF-IDF and Skip-gram Embeddings

This project performs multi-class news topic classification using both traditional machine learning and deep learning approaches. The main goal is to compare the performance of sparse lexical representations such as **TF-IDF** with dense semantic representations such as **Skip-gram Word2Vec embeddings**.

The project evaluates multiple models, preprocessing strategies, and neural architectures to understand why some models perform better than others. The analysis includes accuracy, macro F1-score, class-wise performance, confusion matrices, and training-validation loss curves.

---

## Project Overview

Text classification is a common Natural Language Processing task where a model assigns a category to a given text. In this project, news articles are classified into four categories:

- Business
- Science and Technology
- Sports
- World News

The project investigates how different text representations affect classification performance. Two major representation techniques are compared:

1. **TF-IDF**
   - A sparse lexical representation.
   - Captures important words and n-grams in a document.
   - Useful when class labels are strongly related to specific words or phrases.

2. **Skip-gram Word2Vec**
   - A dense semantic word embedding method.
   - Learns word vectors based on surrounding context.
   - Used with sequential neural models such as RNN, GRU, LSTM, and bidirectional architectures.

---

## Research Motivation

The main motivation of this project was to understand whether traditional lexical features can outperform neural embedding-based models in a news classification task.

The experimental results showed that TF-IDF-based models performed better than Skip-gram-based sequence models. This happened because news category classification is highly vocabulary-driven. Words and phrases such as `stock market`, `world cup`, `software update`, and `prime minister` provide direct signals for their respective classes.

---

## Dataset

The dataset contains news articles belonging to four classes:

| Class | Description |
|---|---|
| Business | News related to markets, companies, economy, finance, and trade |
| Science and Technology | News related to software, internet, research, space, and technology |
| Sports | News related to matches, teams, tournaments, players, and competitions |
| World News | News related to politics, government, international events, and global affairs |

---

## Preprocessing Strategies

Three preprocessing settings were tested:

### 1. No Preprocessing

The raw text was used with minimal cleaning. This preserved all original words and signals, but also kept noise such as rare tokens, punctuation, and unnecessary text fragments.

### 2. Optimum Preprocessing

A balanced preprocessing strategy was used to remove noise while preserving important class-specific information. This setting gave the best average macro F1-score across all models.

### 3. Extreme Preprocessing

A stronger cleaning process was applied. This reduced noise but also risked removing useful information such as named entities, phrase structure, or meaningful terms.

---

## Text Representation Techniques

### TF-IDF

TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures the importance of a word or phrase in a document relative to the whole corpus.

TF-IDF helped the models because news topics are often identified by specific words and phrases.

Example:

| TF-IDF Feature | Likely Class |
|---|---|
| stock market | Business |
| oil prices | Business |
| world cup | Sports |
| football match | Sports |
| software update | Science and Technology |
| search engine | Science and Technology |
| prime minister | World News |
| foreign policy | World News |

The project used TF-IDF with up to 8000 features. This allowed the models to directly learn from thousands of useful lexical and n-gram indicators.

---

### Skip-gram Word2Vec

Skip-gram is an architecture inside Word2Vec. It learns word vectors by using a center word to predict surrounding context words.

Example:

```text
Sentence: oil prices rise in global market

Center word: prices
Context words: oil, rise, global, market
```

In this project, Skip-gram embeddings were trained using:

```python
Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=15
)
```

Here:

- `vector_size=100` means each word was represented by a 100-dimensional vector.
- `window=5` means nearby words within a window of 5 were considered.
- `sg=1` means Skip-gram was used.
- `epochs=15` means Word2Vec training could run for 15 passes over the text.

The embedding vocabulary was limited to the top 10,000 words for neural sequence models. This reduced vocabulary noise and controlled memory usage.

---

## Models Used

The project compared multiple machine learning and deep learning models.

### TF-IDF Based Models

- Logistic Regression
- Deep Neural Network

### Skip-gram Based Sequence Models

- SimpleRNN
- GRU
- LSTM
- Bidirectional SimpleRNN
- Bidirectional GRU
- Bidirectional LSTM

---

## Model Architecture Summary

### TF-IDF + Logistic Regression

```text
Text
-> TF-IDF Vector
-> Logistic Regression
-> Class Prediction
```

Logistic Regression performed strongly because TF-IDF made the dataset mostly linearly separable. Each word or n-gram feature received a class-specific weight.

---

### TF-IDF + Deep Neural Network

```text
Text
-> TF-IDF Vector
-> Dense Layer
-> Dropout
-> Dense Layer
-> Dropout
-> Dense Layer
-> Softmax Output
```

The Deep Neural Network performed best overall because it could learn nonlinear combinations of TF-IDF features.

Example:

```text
apple + shares -> Business
apple + software -> Science and Technology
```

---

### Skip-gram + RNN/GRU/LSTM

```text
Text
-> Tokenization
-> Padded Sequence
-> Skip-gram Embedding Matrix
-> RNN / GRU / LSTM
-> Dense Softmax Output
```

The Skip-gram models used dense word embeddings and sequence learning. These models had to infer the class indirectly from word vectors and word order.

---

## Early Stopping

Early stopping was used in neural network models to prevent overfitting.

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

This means:

- The model monitored validation loss.
- If validation loss did not improve for 3 consecutive epochs, training stopped.
- The best model weights were restored.

Although the maximum number of epochs was set to 15, many models stopped around epoch 3 or 4 because validation loss stopped improving.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro F1-score
- Precision
- Recall
- Class-wise F1-score
- Confusion matrix
- Training and validation loss curves

---

## Accuracy vs Macro F1-score

### Accuracy

Accuracy measures total correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy is useful, but it can hide weak performance on specific classes.

### Macro F1-score

Macro F1 calculates F1-score separately for each class and then takes the average.

```text
Macro F1 = Average of F1 scores of all classes
```

Macro F1 was important in this project because the goal was to classify all four news categories well, not only the easiest class.

---

## Experimental Results

The best-performing model was:

```text
No Preprocessing + TF-IDF + Deep Neural Network
Accuracy: 0.9165
Macro F1-score: 0.9163
```

The second-best model was:

```text
No Preprocessing + TF-IDF + Logistic Regression
Macro F1-score: 0.9120
```

The worst-performing model was:

```text
No Preprocessing + Skip-gram + SimpleRNN
Accuracy: 0.7076
Macro F1-score: 0.7112
```

---

## Why TF-IDF Performed Best

TF-IDF performed best because this dataset is strongly vocabulary-driven. News categories are often identified by specific words and phrases.

For example:

```text
stock market -> Business
world cup -> Sports
software update -> Science and Technology
prime minister -> World News
```

TF-IDF preserved these words and n-grams as direct input features. The classifier could directly connect these features to class labels.

Skip-gram, on the other hand, created dense word vectors based on context. The sequence models then had to infer the class from compressed word embeddings. This made classification harder for this dataset.

---

## Why Logistic Regression Performed Strongly

Logistic Regression performed second best because TF-IDF created a high-dimensional feature space where the classes were mostly separable.

Although Logistic Regression is a linear model, it worked with up to 8000 TF-IDF features. In this high-dimensional space, class-specific words and phrases helped the model separate the four categories effectively.

---

## Why SimpleRNN Performed Worst

The worst model was Skip-gram with SimpleRNN under no preprocessing.

It performed poorly because:

- No preprocessing kept noisy tokens.
- Skip-gram embeddings were frozen and not adjusted for classification.
- SimpleRNN has no gates to control memory.
- Important topic words could be overwritten by later words.
- Business and Science and Technology had overlapping vocabulary.

The confusion matrix showed that the model especially struggled to separate Business and Science and Technology.

---

## Confusion Matrix Analysis

The best model had most predictions on the diagonal of the confusion matrix, meaning it correctly classified most samples in every class.

The worst model had many off-diagonal errors, especially:

```text
Science and Technology -> Business
Business -> Science and Technology
World News -> Business
```

This showed that the worst model had difficulty separating classes with overlapping vocabulary.

---

## Key Findings

- TF-IDF-based models outperformed Skip-gram-based sequence models.
- TF-IDF worked well because news classification depends heavily on class-specific words and phrases.
- Logistic Regression performed strongly because TF-IDF made the task mostly linearly separable.
- Deep Neural Network with TF-IDF performed best because it learned nonlinear combinations of lexical features.
- SimpleRNN performed worst because it had weak memory and no gates.
- GRU and LSTM performed better than SimpleRNN because they use gates to control what information to remember and forget.
- Bidirectional GRU and Bidirectional LSTM performed best among Skip-gram models.
- Optimum preprocessing gave the best average macro F1-score across all models because it balanced noise removal and information preservation.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Gensim
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

---

## Project Files

```text
├── 440_project_FINAL_with_random_tuning_loss_curves.ipynb
├── README.md
└── results/
    ├── confusion_matrices
    ├── loss_curves
    └── model_comparison_tables
```

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/news-classification-tfidf-skipgram.git
```

2. Open the notebook:

```bash
440_project_FINAL_with_random_tuning_loss_curves.ipynb
```

3. Install required libraries if needed:

```bash
pip install numpy pandas scikit-learn tensorflow gensim matplotlib seaborn
```

4. Run all cells in order.

---

## Future Work

Possible future improvements include:

- Testing FastText embeddings for better rare-word handling.
- Using trainable embeddings instead of frozen Skip-gram embeddings.
- Adding attention mechanism with BiGRU or BiLSTM.
- Comparing results with transformer models such as DistilBERT or MiniLM.
- Performing cross-dataset evaluation.
- Measuring model efficiency using training time, inference time, and parameter count.

---

## Conclusion

This project shows that traditional lexical representations can still be highly effective for news classification. TF-IDF outperformed Skip-gram-based sequence models because the dataset was strongly vocabulary-driven. The best model was TF-IDF with a Deep Neural Network, while Logistic Regression also performed strongly due to the separability of TF-IDF features.

The results demonstrate that model performance depends not only on architecture complexity, but also on how well the representation matches the nature of the dataset.
