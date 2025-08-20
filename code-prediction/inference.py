from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
import torch.nn.functional as F

filtered_evidences = pd.read_csv('filtered_evidences.csv')
label_id_df = filtered_evidences.groupby(['label','label_id']).size().reset_index().rename(columns={0:'count'})
labels = label_id_df['label'].tolist()
label_ids = label_id_df['label_id'].tolist()

with open('../code_evidences.json') as sample_desc_file:
    code_evidences = json.load(sample_desc_file)

descriptions = []
for label in tqdm(labels):
    descriptions.append(code_evidences[label][0])

label_id_df['description'] = descriptions

# Define the path to your fine-tuned model
model_path = "./multiclass_icd_finetuned" 

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the model to device type
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set model to evaluation mode
model.eval()

def classify_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    
    predicted_class = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0, predicted_class].item()    
    
    return predicted_class, predicted_prob


# For classifying many text snippets at once
def classify_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze().tolist()

    return probs


sample_text = "prior deep vein thrombosis"
predicted_class, predicted_prob = classify_text(sample_text)

# Convert class index to actual label
predicted_label = labels[label_ids.index(predicted_class)]
predicted_desc = descriptions[label_ids.index(predicted_class)]
print(f"Predicted Clinical Code: {predicted_label}, {predicted_desc}")
