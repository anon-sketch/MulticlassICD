# Multiclass ICD Classification with Silver Evidences

This repository is divided into two main parts:  

1. **Data Creation** – extracting and augmenting evidence spans from MIMIC-IV to create a silver evidence dataset.  
2. **Training** – fine-tuning a classifier on the silver evidence dataset for multiclass ICD prediction.  

---

## 📘 Data Creation  

We extract **evidence spans** from **MIMIC-IV** to build a silver evidence dataset.  

### Prerequisites  
- Access to [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV Note](https://physionet.org/content/mimic-iv-note/2.2/).  
- Access requires completion of the official training (free, ~2–3 hours).  

### Input Format  
After obtaining the data, align text, metadata, codes, and descriptions into a single CSV (or lightweight format). Example format:  

```
Data columns (total 10):
 #   Column                 Non-Null Count   Dtype 
---  ------                 --------------   ----- 
 0   note_id                122300 non-null  object
 1   subject_id             122300 non-null  int64 
 2   hadm_id                122300 non-null  int64 
 3   note_type              122300 non-null  object
 4   note_seq               122300 non-null  int64 
 5   charttime              122300 non-null  object
 6   storetime              122285 non-null  object
 7   text                   122300 non-null  object
 8   icd_codes_list         122300 non-null  object
 9   original_descriptions  122300 non-null  object
```

### Running the Pipeline  
1. Update the CSV file path in **line 50** of `runner.py`.  
2. Run:  
   ```bash
   cd data-creation
   python3 runner.py 'mimic-iv'
   ```  

This will generate the **silver evidence dataset**, stored in JSON format, separated by code type, in the main repository.  

---

## 🔄 Rare Code Augmentation  

We used **GPT-4o-mini** in batches to augment rare codes (you can use any LLM).  

### Prompt Used  
```text
You are given:
- A list of ICD-10 medical codes and their descriptions.
- For each code, a list of existing short evidence phrases.

Your task:
1. For each code, generate exactly 50 additional evidence phrases that could support the given code description.
2. Each evidence must be:
   - Short (3–12 words).
   - Clinically relevant to the code description.
   - Different from all existing evidences for that code (no duplication or trivial rewording).
   - Grammatically correct and natural.
3. Avoid adding unrelated symptoms or conditions.
4. Keep the style consistent with the given examples.

**Output format:**
Return a JSON object where each code maps to an array of exactly 50 new evidence phrases.
```

---

## ✅ Evidence Verification  

In our paper, we conducted **verification studies** to assess the quality of silver evidences:  
- **Model-based Verification**: trained PLM-ICD on silver evidences (adapted from [Edin et al.](https://github.com/JoakimEdin/medical-coding-reproducibility)).  
- **Human-based Verification**: compared against human-annotated evidence spans (see paper for details).  

---

## ⚙️ Training  

Switch to the **code prediction repo** and follow these steps:  

1. Preprocess the data:  
   ```bash
   jupyter notebook preprocess.ipynb
   ```  

2. Train the classifier (modify hyperparameters as needed):  
   ```bash
   python3 classifier.py
   ```  

   The trained model will be saved in the same folder.  

---

## 🔮 Inference  

Run inference using:  
```bash
python3 inference.py
```  

---

✨ That’s it. You now have a multiclass ICD classifier trained on silver evidences!
