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