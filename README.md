# Fake News Detection using Machine Learning

## Overview

This project aims to classify news articles as **Fake News** or **Real News** using Natural Language Processing (NLP) and Machine Learning techniques. It combines data analysis, statistical exploration, text preprocessing, feature extraction and classification models to analyze news content and evaluate the performance of multiple machine learning algorithms.

The project uses **Pandas** and **NumPy** for data analysis, **Matplotlib** and **Seaborn** for visualization, and **Scikit-learn** for machine learning model development and evaluation.

## Objectives

- Perform data cleaning and preprocessing using Pandas and NumPy.
- Conduct Exploratory Data Analysis (EDA) to understand dataset characteristics.
- Analyze statistical patterns and distributions in fake and real news articles.
- Apply Natural Language Processing techniques to transform textual data.
- Extract text features using TF-IDF Vectorization.
- Train and compare multiple machine learning classification models.
- Evaluate models using standard classification metrics.
- Identify the best-performing model based on evaluation results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations and statistical analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| NLTK | Text preprocessing and stemming |
| Scikit-learn | Machine learning and model evaluation |
| Jupyter Notebook | Development and experimentation |

## Dataset

The project uses the **Fake and Real News Dataset**, which contains news articles categorized as fake or real.

The dataset consists of two files:

- `Fake.csv` – Contains fake news articles.
- `True.csv` – Contains real news articles.

Dataset source: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The two datasets are combined and assigned binary labels:

| Label | Category |
|---|---|
| 0 | Real News |
| 1 | Fake News |

## Project Workflow

### 1. Data Collection and Preparation

- Load the fake and real news datasets.
- Assign binary labels.
- Combine datasets into a single DataFrame.
- Shuffle the dataset.
- Inspect dataset structure and data types.

### 2. Data Cleaning and Analysis

Using Pandas and NumPy:

- Handle missing values.
- Detect and remove duplicate records.
- Examine data types and dataset dimensions.
- Generate descriptive statistics.
- Analyze numerical features and distributions.

### 3. Exploratory Data Analysis (EDA)

Exploratory analysis is performed to understand the dataset and discover patterns.

- Distribution of fake and real news articles.
- Article word count analysis.
- Character count analysis.
- Statistical summaries of article lengths.
- Visual comparisons between fake and real news.

### 4. Statistical Analysis

Statistical techniques are used to investigate differences between fake and real news articles.

- Descriptive statistics.
- Mean, median, variance and standard deviation.
- Welch's independent two-sample t-test.
- Mann-Whitney U test.
- Effect size analysis using Cohen's d.

These analyses help investigate whether measurable characteristics of fake and real news articles differ statistically.

### 5. Text Preprocessing

Text preprocessing is performed to prepare news articles for machine learning.

- Lowercase conversion.
- Removal of unnecessary characters and punctuation.
- Tokenization.
- Stopword removal.
- Stemming.

### 6. Feature Engineering

Textual data is transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF helps represent the importance of words in individual documents relative to the overall corpus.

The feature extraction process includes:

- Unigram features.
- Bigram features.
- TF-IDF vectorization.

### 7. Machine Learning Models

The following machine learning algorithms are trained and evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification algorithm |
| Multinomial Naive Bayes | Probabilistic classifier suitable for text classification |
| Linear Support Vector Machine | Margin-based classification algorithm |
| Random Forest | Ensemble learning algorithm based on decision trees |
| SGD Classifier | Linear classifier trained using stochastic gradient descent |

### 8. Model Evaluation and Comparison

Models are evaluated using the following metrics:

- **Accuracy:** Percentage of correctly classified articles.
- **Precision:** Proportion of predicted positive classifications that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes.

The results are presented in a comparative table and visualization to identify the best-performing model.

### 9. Model Interpretation

- Confusion matrix analysis.
- Classification reports.
- Comparison of training and test performance.
- Analysis of informative words using Logistic Regression coefficients.

### 10. News Prediction

The project includes a prediction function that accepts a news article as input and predicts whether the article is classified as Fake or Real using the trained model.

## Model Comparison

The performance of the five machine learning models is compared to evaluate their effectiveness in fake news classification.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9864 | 0.9901 | 0.9838 | 0.9870 | 0.9990 |
| Multinomial Naive Bayes | 0.9353 | 0.9296 | 0.9480 | 0.9387 | 0.9822 |
| Linear SVC | 0.9945 | 0.9957 | 0.9938 | 0.9948 | 0.9998 |
| Random Forest | 0.9938 | 0.9947 | 0.9934 | 0.9940 | 0.9997 |
| SGD Classifier | 0.9902 | 0.9936 | 0.9876 | 0.9906 | 0.9995 |

The final model comparison is generated automatically when the notebook is executed.

## Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/FakeNews_Enhanced.git
```

2. Navigate to the project directory:

```bash
cd FakeNews_Enhanced
```

3. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

4. Download the dataset from Kaggle.

5. Place `Fake.csv` and `True.csv` in the configured dataset directory.

6. Open the Jupyter Notebook:

```bash
jupyter notebook
```

7. Open `FakeNews.ipynb` and execute the notebook cells sequentially.

Alternatively, upload the notebook to Google Colab and execute it there.

## Results

The project generates:

- Exploratory data analysis visualizations.
- Statistical analysis of fake and real news.
- Statistical hypothesis testing results.
- Machine learning model comparison tables.
- Confusion matrices.
- Classification reports.
- Feature importance and informative word analysis.
- Predictions for custom news articles.

## Limitations

- Machine learning models may learn dataset-specific patterns instead of general misinformation characteristics.
- News classification results depend on the quality and representativeness of the dataset.
- TF-IDF models primarily rely on textual patterns and may not capture contextual meaning or factual accuracy.
- High test accuracy on a benchmark dataset does not necessarily translate to reliable detection of real-world misinformation.

## Future Improvements

- Incorporate transformer-based models such as BERT.
- Use cross-validation for more robust model evaluation.
- Evaluate models on unseen datasets from different news sources.
- Implement a web application for real-time news classification.
- Explore explainable AI techniques for prediction transparency.
