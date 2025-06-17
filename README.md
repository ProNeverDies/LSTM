BI-DIRECTIONAL LSTM BASED QUESTION ANSWERING MODEL FOR CYBERSECURITY WHICH CONTINUOUS IMPROVE ON THE BASIS OF FEEDBACK FROM LLM FALLBACK

An advanced NLP model designed for cybersecurity-based question answering using Bidirectional LSTM and BART. Supports continuous learning based on user feedback.

## 🎯 Objectives

- Provide accurate long-form answers to cybersecurity queries
- Continuously improve via user feedback loop
- Integrate threat intelligence via OWASP and CVE datasets

## 🧠 Features

- Bidirectional LSTM for sequence modeling
- BART for generative QA
- Feedback-driven training loop
- OWASP Top 10 & CVE-based vulnerability integration

## 🛠 Tech Stack

- Python
- TensorFlow, Transformers
- Pandas, NumPy
- CVE, OWASP DB

## 🧪 Sample Use Case

> **Q:** What is SQL Injection and how can it be prevented?  
> **A:** SQL Injection is a code injection technique... *(full answer generated)*  
> **User Feedback:** Incorrect  
> **Improvement Module:** Retrains on corrected pattern

## 🚀 How to Run

```bash
python train_model.py
python inference.py
