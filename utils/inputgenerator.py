import pandas as pd
from urllib.parse import urlparse
import csv
import re


def contains_multiple_strings(value):
    # Check if the value contains multiple strings
    if ',' in value or '\n' in value:
        return True

    # Additional check for specific patterns denoting multiple strings
    # For example, considering patterns with new line characters as separators
    # Add more patterns as needed
    patterns = [
        r'\.\s*\n\s*',  # Pattern for a period followed by a new line
        r';\s*\n\s*',   # Pattern for a semicolon followed by a new line
        # Add more patterns if needed
    ]
    for pattern in patterns:
        if re.search(pattern, value):
            return True

    return False

def clean_csv(input_file, output_file):
    cleaned_rows = []
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if contains_multiple_strings(row['Object']):
                row['Object'] = 'description'
            cleaned_rows.append(row)

    # Write cleaned data to a new CSV file
    fieldnames = ['Date', 'Subject', 'Predicate', 'Object', 'Domain', 'Domain Version', 'Domain Date', 'URI', 'Title', 'Languages']
    with open(output_file, 'w', newline='') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

# # Example usage:
# input_filename = 'updated_2023-11-27_Parsed_Knowledge_Triples copy.csv'  # Replace with your input CSV filename
# output_filename = 'cleaned_output.csv'  # Replace with desired output CSV filename
# #clean_csv(input_filename, output_filename)

import pandas as pd
from urllib.parse import urlparse

def filter_csv(input_csv_file, output_csv_file):
    # Read the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_file)

    # Function to extract the part after '#' for 'Predicate' and 'Object' columns
    def extract_after_hashtag(url):
        if isinstance(url, str):  # Check if the value is a string
            if '#' in url:
                return url.split('#')[-1]
            else:
                parsed_url = urlparse(url)
                path = parsed_url.path
                parts = path.split('/')
                last_part = parts[-1] if parts[-1] != '' else parts[-2] if len(parts) > 1 else parsed_url.netloc
                return last_part
        else:
            return url  # Return the original value if it's not a string

    # Apply the function to clean the 'Predicate' and 'Object' columns
    df['Subject'] = df['Subject'].apply(lambda x: extract_after_hashtag(x))
    df['Predicate'] = df['Predicate'].apply(lambda x: extract_after_hashtag(x))
    df['Object'] = df['Object'].apply(lambda x: extract_after_hashtag(x))

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)

# # Example usage:
# input_filename_ = 'cleaned_output.csv'  # Replace with your input CSV filename
# output_filename_ = 'final_output.csv'  # Replace with desired output CSV filename
# filter_csv(input_filename_, output_filename_)

def filter_csv_by_predicate(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row['Predicate'] == 'domain'] #or row['Predicate'] == 'subClassOf' or row['Predicate'] == 'range']

    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# # Usage example:
# input_filename = 'your_input_file.csv'
# output_filename = 'filtered_output.csv'
# filter_csv_by_predicate(input_filename, output_filename)

def group_records_by_domain(data):
    # Function to handle NaN values and convert to strings
    def join_non_null(x):
        non_null_values = [str(val) for val in x if pd.notnull(val)]
        return ', '.join(non_null_values)

    # Grouping records by 'Domain' and aggregating other columns
    grouped_data = data.groupby('Domain').agg({
        'Subject': lambda x: join_non_null(x),
        'Subject-Parent': lambda x: join_non_null(x),
        'Subject-Synonyms': lambda x: join_non_null(x),
        'Subject-Parent-Synonyms': lambda x: join_non_null(x),
        'Subject-Cleaned': lambda x: join_non_null(x),
        'Predicate': 'first',
        'Object': 'first',
        'Domain Version': 'first',
        'Domain Date': 'first',
        'URI': 'first',
        'Title': 'first',
        'Languages': 'first'
        #'corpus': lambda x: join_non_null(x)
    }).reset_index()

    return grouped_data


def create_subject_matrix(data):
    # Splitting the 'Subject' column and creating dummy variables
    subjects = data['Subject'].str.get_dummies(', ')

    # Combining the dummy variables with the original DataFrame
    matrix = pd.concat([subjects, data[['Predicate', 'Object', 'Domain']]], axis=1)

    return matrix

import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if you haven't already
#nltk.download('wordnet')


import spacy
import re
# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

def clean_subject_labels(data):
    def clean_label(text):
        if isinstance(text, str):  # Check if text is a string
            # Remove prefixes like 'has-' and replace hyphens with spaces
            text = re.sub(r"^has-", "", text)
            text = re.sub(r"^has", "", text)
            text = re.sub(r"^is", "", text)
            text = text.replace('-', ' ')

            # Identify camelCase and insert spaces
            text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

            # Split combined words into separate tokens
            text_tokens = re.findall(r'[A-Za-z]+(?:[A-Z][a-z]+)?', text)
            
            # Keep only the last two tokens if more than two tokens exist
            if len(text_tokens) > 2:
                text_tokens = text_tokens[-2:]
            
            text = ' '.join(text_tokens)
            text = text.lower()
            text = re.sub(r"\b of\b", "", text)
            text = re.sub(r"\b for\b", "", text)
            text = re.sub(r"\b on\b", "", text)
            text = re.sub(r"\b by\b", "", text)
            text = re.sub(r"\b ie\b", "", text)
            text = re.sub(r"^of", "", text)
            text = re.sub(r"^in", "", text)
            return text
        else:
            return text  # Return the input if it's not a string

    # Create a new column 'Subject-Cleaned' and apply clean_label function to populate it
    cleaned_data = data.copy()
    cleaned_data['Subject-Cleaned'] = cleaned_data['Subject'].apply(clean_label)
    cleaned_data['Object-Cleaned'] = cleaned_data['Object'].apply(clean_label)
    return cleaned_data #data #cleaned_data

def get_hypernym(word):
    synsets = wn.synsets(word)
    if synsets:
        # Considering the first synset for simplicity
        hypernyms = synsets[0].hypernyms()
        if hypernyms:
            return hypernyms[0].name().split('.')[0]
    return None

def transform_subject_to_hypernym(data):
    def transform_field(field_name):
        parent_values = []
        for phrase in data[field_name]:
            if isinstance(phrase, str):  # Check if the value is a string
                words = phrase.split()
                if len(words) > 0:
                    head_word = words[-1]  # Consider the last word as the head word
                    hypernym = get_hypernym(head_word.lower())
                    # If no hypernym found for the head word, use the original phrase itself
                    if hypernym is None:
                        parent_values.append(phrase)
                    else:
                        parent_values.append(hypernym)
                else:
                    parent_values.append(phrase)  # If phrase is an empty string, keep it unchanged
            else:
                parent_values.append(phrase)  # If it's not a string, keep the original value
        return parent_values

    data['Subject-Parent'] = transform_field('Subject-Cleaned')
    data['Object-Parent'] = transform_field('Object-Cleaned')
    
    return data


import nltk
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    if isinstance(word, str):  # Check if the input is a string
        words = word.split()  # Split the string into individual words
        found_synonyms = False
        
        # Attempt to find synonyms for each word
        for w in words:
            for syn in wordnet.synsets(w):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name()
                    if isinstance(lemma_name, str):  # Check if lemma_name is a string
                        synonyms.append(lemma_name.lower())  # Convert to lowercase and append
                        found_synonyms = True
        
        # If no synonyms found for the entire phrase, search for synonyms of the last word
        if not found_synonyms and words:
            last_word_synonyms = []
            for syn in wordnet.synsets(words[-1]):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name()
                    if isinstance(lemma_name, str):
                        last_word_synonyms.append(lemma_name.lower())
            
            synonyms.extend(last_word_synonyms)
            
    return ', '.join(set(synonyms))  # Convert to set to remove duplicates and then join into a string

def generate_subject_synonyms(data):
    # Apply get_synonyms function to each label in the 'Subject-Cleaned' column
    data['Subject-Synonyms'] = data['Subject-Cleaned'].apply(get_synonyms)
    data['Subject-Parent-Synonyms'] = data['Subject-Parent'].apply(get_synonyms)
    data['Object-Synonyms'] = data['Object-Cleaned'].apply(get_synonyms)
    data['Object-Parent-Synonyms'] = data['Object-Parent'].apply(get_synonyms)
    
    return data


def convert_to_list(text):
    # Check if the input text is a string representation of a list
    if isinstance(text, str):
        # Split the string by comma and remove extra spaces
        text = re.sub(r'\s*,\s*', ',', text)
        return text.split(',')
    else:
        return text

#PROPERTY CORPUS
def generate_corpus_list_p(data):
    # Convert string representation of lists to actual lists
    data['Subject-Synonyms'] = data['Subject-Synonyms'].apply(convert_to_list)
    data['Subject-Parent-Synonyms'] = data['Subject-Parent-Synonyms'].apply(convert_to_list)

    # Combine 'Subject-Synonyms' and 'Subject-Parent-Synonyms' columns into a single 'corpus' column
    data['corpus'] = data.apply(lambda row: row['Subject-Synonyms'] + row['Subject-Parent-Synonyms'], axis=1)

    # Convert the lists in 'corpus' column to strings
    data['corpus'] = data['corpus'].apply(lambda x: ', '.join(x))

    # Extract unique values from 'corpus' column and convert it to a list
    corpus_list = data['corpus'].unique().tolist()

    return corpus_list

#CLASS CORPUS
def generate_corpus_list_c(data):
    # Convert string representation of lists to actual lists
    data['Subject-Cleaned'] = data['Subject-Cleaned'].apply(convert_to_list)
    #data['Subject-Parent-Synonyms'] = data['Subject-Parent-Synonyms'].apply(convert_to_list)

    # Combine 'Subject-Synonyms' and 'Subject-Parent-Synonyms' columns into a single 'corpus' column
    data['corpus'] = data.apply(lambda row: row['Subject-Cleaned'], axis=1)

    # Convert the lists in 'corpus' column to strings
    data['corpus'] = data['corpus'].apply(lambda x: ', '.join(x))

    # Extract unique values from 'corpus' column and convert it to a list
    corpus_list = data['corpus'].unique().tolist()

    return corpus_list

def filter_csv_by_predicates(csv_file_path, predicate_values):
    filtered_records = []
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if row['Predicate'] in predicate_values:
                filtered_records.append(row)
    
    return filtered_records

def filter_dataframe_by_predicates(df, predicate_values):
    filtered_df = df[df['Predicate'].isin(predicate_values)]
    return filtered_df

def filter_dataframe_by_domain(df, predicate_values):
    filtered_df = df[df['Domain'].isin(predicate_values)]
    return filtered_df

def filter_dataframe_by_object(df, predicate_values):
    filtered_df = df[df['Object'].isin(predicate_values)]
    return filtered_df

# def convert_input(input_dict):
#     key, value = list(input_dict.items())[0]
#     return [[key, value]]

def convert_input(input_dict):
    result = []
    for key, value in input_dict.items():
        result.append([key, value])
    return result

def split_sublist(input_list, threshold):
    result = []
    for sublist in input_list:
        key, value = sublist
        clauses = value.split(', ')
        for i in range(0, len(clauses), threshold):
            result.append([key, ', '.join(clauses[i:i+threshold])])
    return result


from pyvis.network import Network
import os

def visualize_networkx_graphs(graph_list, output_dir='network_visualizations'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, (_, graph) in enumerate(graph_list):
        # Create a pyvis Network instance
        net = Network(notebook=False)
        
        # Add nodes and edges to the network
        net.from_nx(graph)

        # Set the output file path
        output_path = os.path.join(output_dir, f'graph_{i + 1}.html')

        # Save the visualization to an HTML file
        net.save_graph(output_path)

        print(f'Graph {i + 1} visualization saved to: {output_path}')