import pandas as pd
import utils.scraper 
import utils.parser
import utils.inputgenerator

#### RUN SCRAPER
utils.scraper.scraper()
file_path = "files/LOV_vocabs.csv"

#### RUN PARSER
vocabs = pd.read_csv(file_path)
utils.parser.parse_vocabs(vocabs)

#### PREPARE INPUT
input_filename = 'files/2023-11-27_Parsed_Knowledge_Triples.csv'  # Replace with your input CSV filename
output_filename = 'files/cleaned_output.csv'  # Replace with desired output CSV filename
utils.inputgenerator.clean_csv(input_filename, output_filename)
input_filename_ = 'files/cleaned_output.csv'  # Replace with your input CSV filename
output_filename_ = 'files/final_output.csv'  # Replace with desired output CSV filename
utils.inputgenerator.filter_csv(input_filename_, output_filename_)
