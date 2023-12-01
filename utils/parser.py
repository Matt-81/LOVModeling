import pandas as pd
import os
import time
from rdflib import Graph
from rdflib.util import guess_format

# Class to handle the Excel file and relative indexes
class ExcelFile:
    def __init__(self, writer, num=0):
        self.writer = writer
        self.index = 1
        self.num = num + 1

# Parse the given file and add its information to the CSV file
def parser(vocabFolder, date, row, totalCsv, list_):
    try:
        g = Graph()
        format_ = row["Link"].split(".")[-1]
        if format_ == "txt":
            format_ = row["Link"].split(".")[-2]
        format_ = format_.split("?")[0]
        result = g.parse(row["Link"], format=guess_format(format_), publicID=row["prefix"])
    except Exception as e:
        print(str(e) + "\n")
        return totalCsv, list_, 0

    fileName = date + "_Parsed_" + row["prefix"] + "_" + row["VersionName"] + "_" + row["VersionDate"] + ".csv"

    # For each statement present in the graph obtained, store the triples
    for subject, predicate, object_ in g:
        # Here, we create a basic CSV structure representing RDF triples
        list_.append({
            "Date": date,
            "Subject": subject,
            "Predicate": predicate,
            "Object": object_,
            "Domain": row["prefix"],
            "Domain Version": row["VersionName"],
            "Domain Date": row["VersionDate"],
            "URI": row["URI"],
            "Title": row["Title"],
            "Languages": row["Languages"]
        })

    # Save the information to the CSV file
    df = pd.DataFrame(list_)
    csv_file_path = os.path.join(vocabFolder, fileName)
    df.to_csv(csv_file_path, index=False)

    return totalCsv, list_, len(list_)

def parse_vocabs(vocabs):
    folderDestination = "files" #"%{folderDestination}"
    location = os.path.normpath(os.path.expanduser(folderDestination))
    if not os.path.isdir(location):
        os.makedirs(location)

    date = time.strftime("%Y-%m-%d", time.gmtime())

    # Create a DataFrame to store information
    df = pd.DataFrame(columns=[
        "Date", "Subject", "Predicate", "Object",
        "Domain", "Domain Version", "Domain Date",
        "URI", "Title", "Languages"
    ])

    # Iterate for every vocabulary read from the second argument
    frames = []  # List to store individual DataFrames
    for index, row in vocabs.iterrows():
        vocabFolder = str(os.path.join(location, row["Folder"]))
        if not os.path.isdir(vocabFolder):
            os.makedirs(vocabFolder)
        
        # Initialize a list to store triples for each vocabulary
        triples_list = []
        totalCsv, triples_list, i = parser(vocabFolder, date, row, None, triples_list)
        if i and len(triples_list):
            df_temp = pd.DataFrame(triples_list)
            frames.append(df_temp)

    # Concatenate all individual DataFrames into a single DataFrame
    if frames:
        df = pd.concat(frames, ignore_index=True)

    # Save the information to a single CSV file
    total_csv_file_path = os.path.join(location, date + "_Parsed_Knowledge_Triples.csv")
    df.to_csv(total_csv_file_path, index=False)

    return df