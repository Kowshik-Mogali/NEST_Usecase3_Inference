# Clinical Trial Completion Prediction - Inference Guide

## Introduction

This repository provides the inference pipeline for predicting the completion status of clinical trials using a combination of **fine-tuned ClinicalBioBERT models, Facebook BART LLM for Oncology Classification, and XGBoost models**. The goal is to classify whether a clinical trial will be **completed** or **not completed** (Suspended, Withdrawn, or Terminated) using both **structured** and **unstructured** data.


---

## Directory Structure

Download the `Usecase_3_.csv` file from the following google drive location and keep it in the root directory.
`https://drive.google.com/drive/folders/1pigBb4Lx0YJPBBx9rdh6V2FTUTV713zw?usp=sharing`

```
.
├── inference/                         # Main inference directory
│   ├── ClinicalBioBertFinal/          # Fine-tuned ClinicalBioBERT models for different text fields
│   ├── FacebookBART/                   # Facebook BART model for text embeddings
│   ├── one_hot_encoder.pkl            # Pre-trained OneHotEncoder model for categorical encoding
│   ├── one_hot_encoded_feature_names.json  # Feature names used during training
│   ├── Usecase_3_.csv                  # Test dataset (to be provided by the user)
│   ├── Results.csv                     # Output predictions
│   ├── README.md                       # This file
│   ├── requirements.txt                # Dependencies for setting up the environment
```

---

## 1️⃣ Data Preprocessing and Splitting

### **Data Splitting Strategy**

The dataset is split into **six categories** based on **Study Type** and **Conditions Category**:

1. **Interventional - Oncology**  
2. **Interventional - Non-Oncology**  
3. **Interventional - Other**  
4. **Observational - Oncology**  
5. **Observational - Non-Oncology**  
6. **Observational - Other**  

Each category has a **separate XGBoost model** trained to predict the completion status, ensuring better accuracy across different trial types.

### **Feature Engineering Steps**
- **Unstructured text data** from **Study Title, Brief Summary, Conditions, Outcomes, etc.** is passed through **Facebook BART LLM** to generate **Conditions Category**.
- **ClinicalBioBERT models** are fine-tuned on text fields to predict individual probabilities.
- **OneHotEncoder** is used for categorical features like **Funding Type, Study Type, and Phase**.
- **XGBoost models** use a combination of numerical, categorical, and text-based embeddings for the final classification.

---

## 2️⃣ Installation Guide

### **Set Up Conda Environment**
```bash
conda env create -f NEST_environment.yml
conda activate NEST
```

Alternatively, install required Python packages using:
```bash
pip install -r requirements.txt
```

---

## 3️⃣ Required Data & Model Files

Ensure the following files and directories are correctly placed inside `inference/`:

✅ **Fine-Tuned ClinicalBioBERT Models** (`inference/ClinicalBioBertFinal/`)  
- Used for text classification on different trial text fields.  

✅ **Pre-Trained OneHotEncoder Files**  
- `one_hot_encoder.pkl` → The saved OneHotEncoder model.  
- `one_hot_encoded_feature_names.json` → Contains categorical feature names used during training.  

✅ **Test Data**  
- Paste your test dataset inside `inference/`  
- Example: `inference/Usecase_3_.csv`  

---

## 4️⃣ Running Inference
Run Inference.ipynb notebook

### **Steps Performed**

1️⃣ **Load Models & Encoders**  
   - Load **fine-tuned ClinicalBioBERT models** for text classification.  
   - Load **Facebook BART model** to generate text embeddings.  
   - Load **OneHotEncoder** for categorical feature encoding.  

2️⃣ **Preprocess Test Data**  
   - Tokenize and embed text fields using **Facebook BART**.  
   - One-hot encode categorical features using `one_hot_encoder.pkl`.  
   - Standardize numerical variables for model compatibility.  

3️⃣ **Predict Using XGBoost Models**  
   - Select the appropriate XGBoost model based on **Study Type** and **Condition Category**.  
   - Use text embeddings + structured data for classification.  
   - Apply **majority voting** across different models for the final decision.  

4️⃣ **Post-Processing & Explainability**  
   - Extract feature importance from XGBoost models.  
   - Assign explainability scores to top contributing features.  
   - Save predictions in `Results.csv`.  

---

## 5️⃣ Expected Output

Once inference is complete, predictions will be stored in `inference/Results.csv`.

**Example Output Format:**
| NCT Number | Final Prediction_Max_Voting | Final Prediction (Heuristic) |
|------------|-----------------------------|------------------------------|
| NCT123456  | 1                           | 1                            | 
| NCT789012  | 0                           | 0                            | 

- `Final Prediction_Max_Voting` → Majority voting decision from multiple classifiers.
- `Final Prediction (Heuristic)` → Rule-based prediction using heuristic conditions.

---

## 6️⃣ Performance Metrics

After running inference, performance can be evaluated using:

- **Precision, Recall, F1-score** (for Completed vs. Not Completed)
- **Confusion Matrix**
- **AUC-ROC Score** (Measures classification performance)
- **Explainability Score** (Feature impact weights)

---

## 7️⃣ Next Steps

- **Fine-tune more ClinicalBioBERT models** for improved classification.
- **Enhance XGBoost models** with hyperparameter tuning.
- **Explore Causal Inference Techniques** for better explainability.

For any issues, please raise an issue in this repository. 🚀
