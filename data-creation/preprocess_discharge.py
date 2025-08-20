from tqdm import tqdm
import string
import re
import json
from nltk.corpus import stopwords

# List of predefined sections
# REST keyword is used to indicate everything except these sections
SECTIONS = ['Complaint', 'Major Surgical or Invasive Procedure', 'History of Present Illness', 'Past Medical History', 'Family History', 'Social History', 'Other History', 'Diagnosis', 'Brief Hospital Course', 'PMH', 'HPI', 'EGD:', 'REST']

with open('lookup_alphabetical.json', 'r') as lookup_file:
    lookup_json = json.load(lookup_file)
LOOKUP_DICT = {item['key']: item['value'] for item in lookup_json}


def main_preprocess(text, search_keyword):
    """
    Preprocess the discharge text based on the given section keyword.
    """
    # If keywords are in one of these sections, combine linebreaks and split on the basis of given punctuation marks (. , :)
    if search_keyword in ['Brief Hospital Course', 'History of Present Illness', 'Major Surgical or Invasive Procedure', 'Family History']:
        linebreak_indices = [i for i, letter in enumerate(text) if letter == '\n']
        for index in linebreak_indices:
            if index != 0 and text[index - 1] not in ['\n', '.']:
                text = text[:index] + ' ' + text[index + 1:]

    text_splits = text.lower().split('\n')
    text_splits = [text.strip() for text in text_splits if len(text.strip()) > 1]

    text = '.'.join(text_splits)

    # Remove content inside [**..**]
    text = re.sub(r'\[.*?\]|\*\*.*?\*\*', '', text)

    punctuation_without_dot = string.punctuation.replace("/", "").replace("'", "").replace(".", "")

    if search_keyword in ['Past Medical History', 'PMH']:
        punctuation_without_dot = punctuation_without_dot.replace(",","")

    if search_keyword == 'Brief Hospital Course':
        punctuation_without_dot = punctuation_without_dot.replace(":","")

    # Replace punctuation (except slash, apostrophe, and dot) with whitespace
    text = text.translate(str.maketrans(punctuation_without_dot, ' ' * len(punctuation_without_dot)))

    # Adding space around remaining punctuation marks
    text = re.sub(r"(/'.:,)", r" \1 ", text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split dots unless they are between numbers
    pattern = r'(?<!\d)\.|(?<!\d)\.(?!\d)|\.(?!\d)'
    segments = re.split(pattern, text)

    if search_keyword in ['Past Medical History', 'PMH', 'PSH']:
        segments = [segment for seg in segments for segment in seg.split(',')]
    
    if search_keyword == 'Brief Hospital Course':
        segments = [segment for seg in segments for segment in seg.split(':')]
    
    # Strip extra spaces and replace words using lookup dictionary
    final_segments = []
    for segment in [segment.strip() for segment in segments if len(segment.strip()) > 1]:
        words = segment.split(' ')
        replaced_words = [LOOKUP_DICT.get(word, word) for word in words]
        final_segments.append(" ".join(replaced_words))

    return final_segments


def get_sections(discharge_text, search_keyword):
    """
    Extract and preprocess a specific section from the discharge text.
    Most easy option is just send ALL keyword, it will preprocess everything at once.
    For finer detailing, we separate into specific sections.
    """
    if search_keyword == 'ALL':
        return main_preprocess(discharge_text, search_keyword)

    if search_keyword == 'REST':
        final_split = discharge_text.split('Family History')[-1].split('Discharge Diagnosis')
    else:
        splits = discharge_text.split(search_keyword)

        final_split = ['']
        candidate_split_item = splits[-1]
        # Assuming splits is a list of lists or strings
        if not candidate_split_item:  # Check if candidate_split_item is empty
            for next_split in splits[1:]:  # Start looking from the second element
                if next_split:  # Check if the current element is not empty
                    candidate_split_item = next_split  # Assign it to candidate_split_item
                    break

        if len(splits) > 1:
            if search_keyword == 'Complaint':
                split_keywords = ['Major Surgical or Invasive Procedure', 'HPI', 'History of Present Illness', 'Past Medical History', 'PMH', 'Family History', 'Social History']
                occurrences = [(candidate_split_item.find(k), k) for k in split_keywords if k in candidate_split_item]
                if occurrences:
                    first_occurrence = min(occurrences)
                    final_split = candidate_split_item.split(first_occurrence[1])
                else:
                    final_split = candidate_split_item.split('\n') 

            elif search_keyword == 'Major Surgical or Invasive Procedure':
                final_split = candidate_split_item.split('History of Present Illness')
            
            elif search_keyword == 'History of Present Illness':
                split_keywords = ['Past Medical History', 'Family History', 'Social History']
                occurrences = [(candidate_split_item.find(k), k) for k in split_keywords if k in candidate_split_item]
                if occurrences:
                    first_occurrence = min(occurrences)
                    final_split = candidate_split_item.split(first_occurrence[1])
                else:
                    final_split = candidate_split_item.split('PMH')
            
            elif search_keyword in ['Past Medical History', 'PMH']:
                split_keywords = ['Family History', 'Social History', 'PSH', 'Review']
                # Find the first occurring keyword
                occurrences = [(candidate_split_item.find(k), k) for k in split_keywords if k in candidate_split_item]
                if occurrences:
                    first_occurrence = min(occurrences)
                    final_split = candidate_split_item.split(first_occurrence[1])
                else:
                    final_split = [candidate_split_item] 

            elif search_keyword == 'Family History':
                split_keywords = ['Social History','Physical Exam']
                # Find the first occurring keyword
                occurrences = [(candidate_split_item.find(k), k) for k in split_keywords if k in candidate_split_item]
                if occurrences:
                    first_occurrence = min(occurrences)
                    final_split = candidate_split_item.split(first_occurrence[1])
                else:
                    final_split = [candidate_split_item] 

            elif search_keyword == 'Social History':
                final_split = candidate_split_item.split('Family History')
                if len(final_split) == 1:
                    final_split = candidate_split_item.split('Physical Exam')

            elif search_keyword == 'Brief Hospital Course':
                final_split = candidate_split_item.split('Medications')
            elif search_keyword == 'Diagnosis:':
                final_split = candidate_split_item.split('Discharge Condition')
            else:
                final_split = candidate_split_item.split('\n \n')
            
        else:
            # Sometimes sections don't adhere to cases, so we convert everything to lowercase and check again
            splits = (discharge_text.lower()).split(search_keyword.lower())
            all_search_keywords = [keyword.lower().translate(str.maketrans('', '', string.punctuation)) for keyword in SECTIONS]
            if len(splits) > 1:
                candidate_split_item = splits[-1]
                occurrences = [(candidate_split_item.find(k), k) for k in all_search_keywords if k in candidate_split_item]
                if occurrences:
                    first_occurrence = min(occurrences)
                    final_split = candidate_split_item.split(first_occurrence[1])
                else:
                    final_split = [candidate_split_item] 

            else:
                return ''

    return main_preprocess(final_split[0], search_keyword)


def preprocess_discharge_df(icd_10_discharges):
    """
    Preprocess discharge data and extract diagnosis entries.
    """
    discharge_texts = icd_10_discharges['text'].tolist()
    hadm_ids = icd_10_discharges['hadm_id'].tolist()
    subject_ids = icd_10_discharges['subject_id'].tolist()
    icd_codes = icd_10_discharges['icd_codes_list'].tolist()
    original_descriptions = icd_10_discharges['original_descriptions'].tolist()

    diagnosis_entries = []

    for i, discharge in enumerate(tqdm(discharge_texts, desc="Preprocessing Discharge Notes")):
        new_entry = [get_sections(discharge, section) for section in SECTIONS]

        # If all sections are empty, process the entire text
        if len(new_entry[1:]) == new_entry[1:].count(''):
            new_entry.append(get_sections(discharge, 'ALL'))

        flattened_evidence = [evidence for evidences in new_entry for evidence in evidences]

        stop_words = set(stopwords.words('english'))
        new_flattened_evidence = []
        temp_evidences = []

        for evidence in flattened_evidence:
            evidence_words = evidence.split()
            filtered_evidence_words = [word for word in evidence_words if word not in stop_words]

            # Join the filtered words back into a single string
            new_evidence = " ".join(filtered_evidence_words)

            if new_evidence not in temp_evidences:
                temp_evidences.append(new_evidence)
                new_flattened_evidence.append(evidence)

        diagnosis_entries.append([
            subject_ids[i],
            hadm_ids[i], 
            new_flattened_evidence, 
            icd_codes[i], 
            original_descriptions[i],
        ])

    return diagnosis_entries