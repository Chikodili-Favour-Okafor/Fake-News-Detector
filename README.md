# Fake News Detection Using Machine Learning

## Project Overview

The spread of misinformation through digital platforms makes it increasingly difficult for readers to distinguish reliable news from misleading information. This project applies Natural Language Processing (NLP) and machine learning to classify news articles as **FAKE** or **REAL** based on their textual content.

Two classification algorithms, Logistic Regression and Linear Support Vector Machine (SVM), were developed and evaluated. Linear SVM was selected as the final model because it achieved better performance on the held-out test dataset.

The selected model was integrated into a Streamlit web application, allowing users to enter news headlines and article text, obtain predictions, and review their prediction history.

> **Disclaimer:** This application is an experimental machine-learning classifier, not an independent fact-checking system. Its predictions should not be treated as definitive proof that a news article is true or false.

## Project Objectives

- Clean and prepare news article data for machine learning.
- Explore the dataset and examine its quality and class distribution.
- Combine news headlines and article bodies to create a unified text feature.
- Apply TF-IDF vectorization to transform textual data into numerical features.
- Train and compare Logistic Regression and Linear SVM.
- Evaluate model performance using appropriate classification metrics.
- Save the selected model as a reusable machine-learning pipeline.
- Develop and deploy an interactive Streamlit application for news classification.

## Technologies and Tools

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Natural Language Processing:** Scikit-learn TF-IDF Vectorizer
- **Machine Learning:** Logistic Regression, Linear SVM
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## Dataset

The project uses the Fake News Dataset, stored locally as `fake_or_real_news.csv`.

The original dataset contains the following columns:

| Column | Description |
|---|---|
| `id` | Article identifier |
| `title` | News headline |
| `text` | News article body |
| `label` | Target class: `FAKE` or `REAL` |

The `title` and `text` columns provide the textual information used for classification, while `label` is the target variable.

The dataset was cleaned before model training. The final dataset contained approximately 6,000 articles after removing records with missing article text and duplicate text entries. The exact number of records retained should be confirmed from the final cleaned dataset.

## Project Workflow

### 1. Data Collection and Understanding

The dataset was loaded into Pandas for inspection. The initial analysis examined the dataset's structure, data types, missing values, class distribution, and duplicate records.

### 2. Data Cleaning

The following data-quality checks and cleaning operations were performed:

- Examined missing values in the dataset.
- Removed rows that contained no useful information.
- Removed records without article text, since the text was required for classification.
- Identified and removed duplicate article text entries.
- Checked for conflicting labels among duplicated texts.
- Examined text encoding issues and unusual data types.

These steps helped improve data quality and reduce unnecessary duplication before model development.

### 3. Combining the Headline and Article Body

A new column called `model_text` was created by concatenating the headline and article body.

This allowed the model to use information from both sources instead of relying on the headline or article body alone.

Conceptually:

`model_text = title + text`

Missing headlines were handled by replacing them with empty strings during concatenation.

### 4. Text Preprocessing

A custom text preprocessing function was developed and saved in `text_preprocessing.py`.

The preprocessing steps include:

- Converting text to lowercase.
- Removing URLs.
- Removing HTML tags.
- Normalizing whitespace and trimming unnecessary spaces.

The function is incorporated into the saved machine-learning pipeline, ensuring that new articles submitted through the application receive the same preprocessing as the model's input data.

### 5. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split and stratified sampling.

- **Training set:** 4,831 articles
- **Testing set:** 1,208 articles

The split used `random_state=42` to support reproducibility and stratification to preserve the approximate class proportions in both sets.

The TF-IDF vectorizer was fitted on the training text only and then applied to the test text. This helps prevent information leakage from the test set into feature learning.

### 6. Feature Engineering with TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert the preprocessed news text into numerical features suitable for machine learning.

The vectorizer was configured with:

- `max_features=50000`: Keep up to 50,000 selected features.
- `ngram_range=(1, 2)`: Consider individual words and consecutive two-word combinations.
- `min_df=2`: Ignore features appearing in fewer than two training documents.
- `max_df=0.95`: Ignore features appearing in more than 95% of training documents.

The fitted vectorizer learns its vocabulary and IDF weights from the training data and reuses them when processing test articles and new application inputs.

### 7. Model Development

Two machine-learning algorithms were trained and evaluated.

**Logistic Regression**

Logistic Regression was used as the baseline classifier. It provided an initial performance benchmark against which the second model could be compared.

**Linear Support Vector Machine (SVM)**

Linear SVM was trained using the same training data and TF-IDF feature representation. It was selected as the final model because it achieved better results on the held-out test dataset.

The target labels remained as `FAKE` and `REAL`, since both Scikit-learn classifiers support string class labels without requiring manual label encoding.

## Model Evaluation

The models were evaluated on the held-out test dataset using accuracy, precision, recall, F1-score, and a confusion matrix.

The reported results were:

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 91% | 93% |
| FAKE Precision | 89% | 92% |
| FAKE Recall | 94% | 95% |
| FAKE F1-score | 91% | 94% |

*Values are rounded to two decimal places where appropriate.*

### Why Linear SVM Was Selected

Linear SVM achieved 93% accuracy compared with 91% for Logistic Regression. It also achieved higher precision and recall for the FAKE class, producing a stronger balance between identifying fake news and limiting false alarms on the test set.

For this project, `FAKE` was treated as the positive class when calculating precision, recall, and F1-score.

The final model's confusion matrix, using class order `[FAKE, REAL]`, was:

| Actual / Predicted | Predicted FAKE | Predicted REAL |
|---|---:|---:|
| Actual FAKE | 582 | 30 |
| Actual REAL | 50 | 546 |

This means:

- 582 fake articles were correctly classified as FAKE.
- 30 fake articles were incorrectly classified as REAL.
- 50 real articles were incorrectly classified as FAKE.
- 546 real articles were correctly classified as REAL.

These results describe performance on the test dataset and do not guarantee equivalent performance on articles from new sources or different time periods.

## Model Saving and Reusability

The selected model was saved as:

`fake_news_pipeline.joblib`

The saved pipeline contains the fitted text preprocessing transformer, TF-IDF vectorizer, and Linear SVM classifier.

This approach makes the model easier to reuse because text preprocessing, feature transformation, and classification can be performed through a single prediction interface.

The custom preprocessing function is maintained in `text_preprocessing.py`, which must be available when loading the saved pipeline.

## Streamlit Web Application

An interactive Streamlit application was developed to make the trained model accessible through a web interface.

### Application Features

- **News Input:** Enter a headline and article body.
- **Sample Article:** Populate the input fields with an example for testing.
- **Article Classification:** Submit text and receive a FAKE or REAL prediction.
- **Reset Inputs:** Clear the current headline and article body.
- **Model Performance Dashboard:** Display previously measured test metrics.
- **Prediction History:** Review predictions made during the current session.
- **CSV Download:** Export the session's prediction history for further analysis.
- **Model Information:** View information about the classifier and its evaluation.

The performance metrics displayed in the dashboard represent the model's earlier evaluation results. They are not recalculated for every new article.

The application does not independently verify news claims or establish the factual accuracy of individual articles.

## Deployment

The application was prepared for deployment through Streamlit Community Cloud using GitHub for version control and source-code hosting.

The deployment repository contains the application code, custom preprocessing module, saved model pipeline, dependency file, and project documentation.

The deployed application loads the saved pipeline and processes submitted articles without requiring the user to run the training notebook or provide the original training dataset.

### Running the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Fake_News_Detection.git
cd Fake_News_Detection
```

Replace the example repository URL with the actual GitHub repository URL.

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For reproducible deployment, use compatible dependency versions, particularly the version of Scikit-learn used to save the model.

**5. Run the application**

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`.

## Project Structure

```text
Fake_News_Detection/
├── app.py
├── text_preprocessing.py
├── fake_news_pipeline.joblib
├── Fake News .ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/                  # Local environment; not uploaded
```

The notebook documents the model development process. The application, preprocessing module, saved model, and dependency file support inference and deployment.

The original dataset is not required to run the deployed application, provided the saved pipeline and its dependencies are available.

## Limitations

- The model learns patterns from its training dataset and may not generalize to all publishers, topics, writing styles, or future events.
- TF-IDF represents text using word and word-pair features but does not independently establish factual truth or understand every aspect of context.
- The model can incorrectly classify both fake and genuine articles.
- The current application returns a class prediction rather than a calibrated probability of truth.
- The current evaluation reflects one stratified train-test split; additional external and time-based testing would provide stronger evidence of generalization.
- Prediction history is session-based rather than stored in a persistent database.

## Future Improvements

### 1. Advanced NLP and Transformer Models

Explore transformer-based language models and other contextual text representations to determine whether they improve classification performance beyond the current TF-IDF and Linear SVM approach.

### 2. Agentic AI Engineering

Explore **agentic AI engineering** to advance the project beyond single-model text classification. The goal is to investigate a controlled workflow in which AI agents or agentic components perform complementary tasks, such as extracting claims from articles, retrieving evidence from trusted sources, comparing claims against supporting information, and generating evidence-based explanations.

The aim is to improve the reliability and usefulness of the system through evidence retrieval, source verification, and coordinated reasoning, rather than simply changing the classifier. Any proposed agentic approach will require careful evaluation against the current model, including factual accuracy, false-positive and false-negative rates, latency, cost, and the reliability of cited evidence.

### 3. Improved Data Quality and Coverage

Expand and update the dataset with diverse publishers, topics, time periods, and writing styles. Investigate duplicate leakage, source bias, and label quality to support more reliable evaluation.


### 4. Explainability and Confidence Calibration

Investigate methods for explaining influential textual features and, if useful, calibrating model scores. Any confidence or uncertainty display should be validated and must not be presented as proof of factual truth.

### 5. Evidence-Based Fact-Checking

Integrate trusted news sources, official announcements, and reliable reference material to support claim verification. Clearly distinguish between a text-based model prediction and a conclusion supported by external evidence.

### 6. Application and Deployment Improvements

Introduce persistent prediction history, batch article analysis, stronger input validation, automated testing, model versioning, monitoring, and a more robust deployment workflow.

## Conclusion

This project demonstrates an end-to-end machine-learning workflow, from data cleaning and NLP feature engineering to model comparison, evaluation, pipeline serialization, and web application deployment.

Linear SVM achieved the best performance among the two evaluated models, with 93% test accuracy and a 94% F1-score for the FAKE class. The Streamlit application provides an accessible way to experiment with the model, while the planned future work focuses on stronger generalization, evidence-based verification, and exploring agentic AI engineering.

The project serves as a foundation for further investigation into practical misinformation detection systems.

## Author
Chikodili Favour Okafor

Data Scientist | AI / Machine Learning Enthusiast

GitHub: https://github.com/chikodili-favour-okafor


