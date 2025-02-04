# **Online Job Postings Resume Screening with JobFitNLP**

## **📌 Project Overview**
This project automates **resume screening** using **Natural Language Processing (NLP) and Deep Learning**. It classifies job postings and resumes as **IT or Non-IT jobs**, leveraging **Word2Vec embeddings** and an **LSTM-based deep learning model**.

## **🔍 Key Features**
✅ **Preprocessing**: Cleans job descriptions and resumes using NLP techniques.
✅ **Word2Vec Embeddings**: Transforms text data into meaningful numerical vectors.
✅ **Deep Learning Model**: Trains an LSTM model to classify job postings.
✅ **Resume Screening Function**: Predicts whether a resume is relevant to an IT job.
✅ **92% Accuracy**: Achieved high classification accuracy for IT vs. Non-IT jobs.

---

## **📂 Dataset**
The dataset contains **19,001 job postings** with attributes like:
- **Job Title, Company, Date, Location, Salary, Job Description, Requirements, and Duration.**
- **Missing values handled** through preprocessing.
- **Columns like AnnouncementCode and Audience removed** for better model efficiency.

---

## **📊 Exploratory Data Analysis (EDA)**
📌 **Job Market Trends**: 
- Top 10 job titles and hiring companies analyzed.
- IT vs. Non-IT job distribution visualized.
- Monthly job postings and top job locations plotted.
- Word cloud created to extract **important keywords from job descriptions**.

📌 **Text Preprocessing**:
- Tokenization, Lemmatization, Stopword Removal
- Combining job descriptions and requirements into a single feature
- Cleaning unnecessary characters

---

## **⚙️ Model Architecture**
📌 **Word2Vec Embeddings**
- Trained on job descriptions to generate **300-dimensional vectors**.
- Converts text into **meaningful numerical representations**.

📌 **Deep Learning Model**
- **Architecture**:
  - Dense layers for feature extraction.
  - LSTM layers for sequence processing.
  - Dropout layers for regularization.
  - Sigmoid activation for binary classification (IT vs. Non-IT).
- **Training Details**:
  - **80-20 train-test split**
  - **15 epochs, batch size: 64**
  - **Adam optimizer & binary cross-entropy loss**

📌 **Performance**:
- **Accuracy: 92.38%**
- **Precision, Recall, F1-score calculated**
- **Confusion Matrix shows effective classification**

---

## **🚀 Installation & Setup**
### **🔹 Prerequisites**
Ensure you have **Python 3.7+** installed along with the required libraries.

```bash
pip install numpy pandas matplotlib seaborn gensim scikit-learn tensorflow nltk wordcloud
```

### **🔹 Clone the Repository**
```bash
git clone https://github.com/your-username/JobFitNLP.git
cd JobFitNLP
```

### **🔹 Run the Project**
```python
import pandas as pd
from jobfitnlp import predict_resume

resume = """
Experienced software developer with 5+ years in Python and machine learning.
Proficient in TensorFlow and cloud platforms like AWS.
"""
print(predict_resume(resume))  # Output: IT Job
```

---

## **💡 Future Improvements**
📌 **Upgrade Word2Vec to Transformer-based embeddings (BERT).**
📌 **Deploy as a web application for real-time resume screening.**
📌 **Enhance job matching by incorporating job-specific skills.**

---

## **👨‍💻 Contributing**
We welcome contributions! Feel free to fork the repo and submit **pull requests**. 😊

---

## **📜 License**
This project is open-source under the **MIT License**.

🚀 **Happy Coding!**
