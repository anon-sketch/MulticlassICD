import sys
import pandas as pd
import json

sys.path.append('../')

from preprocess_discharge import preprocess_discharge_df
from mimic_embedding_matching import match_diagnosis

def load_discharge_data(file_path):
    print(f'Loading discharge data from {file_path}')
    return pd.read_csv(file_path)

def preprocess_data(discharge_df):
    return preprocess_discharge_df(discharge_df)

def match_embeddings(preprocessed_data):
    return match_diagnosis(preprocessed_data)


def extract_codes(codes):
    obtained_codes_sets = []    
    for final_codes in codes:
        obtained_codes = []
        for code in final_codes:
            if len(code[0]) > 1:
                multiple_codes = code[0].replace('[', '').replace(']', '').split()
                obtained_codes.extend(multiple_codes)
        obtained_codes_sets.append(obtained_codes)
    return obtained_codes_sets


def save_dict_to_json(file_path, given_dict):
    with open(file_path, 'w') as outfile:
        json.dump(given_dict, outfile, indent=4, ensure_ascii=False)

    print(f'JSON file saved at {file_path}')


def main():
    # Choose file paths
    if len(sys.argv) != 2:
        print("Usage: python runner.py <mdace|mimic-iv>")
        sys.exit(1)

    # Get the argument from the user
    choice = sys.argv[1].lower()

    # File paths
    file_paths = {
        "mimic-iv": "../physionet.org/files/mimic-iv-note/2.2/note/icd_10_discharge.csv",
        "mimic-iii":"../physionet.org/files/mimiciii/1.4/mimiciii_icd_discharge.csv",
    }

    # Select the file path based on user input
    discharge_file_path = file_paths.get(choice)
    file_path_options = file_paths.keys()

    if not discharge_file_path:
        print(f"Invalid choice. Please use {file_path_options}.")
        sys.exit(1)


    output_file_path = '../code_evidences.json'

    # Load and process discharge data
    discharge_df = load_discharge_data(discharge_file_path)
    preprocessed_data = preprocess_data(discharge_df)
    evidence_dict = match_embeddings(preprocessed_data)
    save_dict_to_json(output_file_path, evidence_dict)


if __name__ == '__main__':
    main()

