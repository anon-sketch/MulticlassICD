import ast
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import random
from tqdm import tqdm

# Initialize the sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print('Embedding Model loaded')


def match_diagnosis(preprocessed_data):
    """
    Matches diagnosis descriptions to their most similar evidence texts based on embeddings.

    Args:
        diagnosis_entries (list): A list of diagnosis entries, where each entry is a list containing
                                  HADM ID, evidence texts, ICD codes, descriptions, and annotated evidences.

    Returns:
        str: Path to the output CSV file containing the final diagnosis dataset.
    """
    

    evidences = {}
    for entry in tqdm(preprocessed_data, desc="Embedding Matching with Original Description"):
        flattened_diagnosis_text = entry[2]
        icd_codes = ast.literal_eval(entry[3])
        dataset_diagnosis_descriptions = ast.literal_eval(entry[4])

        # Encode all evidence texts into embeddings
        corpus_embeddings = embedder.encode(flattened_diagnosis_text, convert_to_tensor=True)

        for i, description in enumerate(dataset_diagnosis_descriptions):
            query_embedding = embedder.encode(description, convert_to_tensor=True)

            # We use cosine-similarity to find the top score
            similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
            _, max_idx = torch.max(similarity_scores, dim=0)

            code_evidence_list = evidences.get(icd_codes[i], [description])
            current_evidence = flattened_diagnosis_text[max_idx]
            if current_evidence not in code_evidence_list:
                code_evidence_list.append(current_evidence)

            evidences[icd_codes[i]] = code_evidence_list

    return evidences