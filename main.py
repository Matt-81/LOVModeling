import pandas as pd
import utils.scraper 
import utils.parser
import utils.inputgenerator
import utils.matching

import pandas as pd
import bnlearn as bn

#### RUN SCRAPER
#utils.scraper.scraper()
file_path = "files/LOV_vocabs.csv"

#### RUN PARSER
vocabs = pd.read_csv(file_path)
#print(vocabs)
#utils.parser.parse_vocabs(vocabs)

#### PREPARE INPUT
input_filename = 'files/2023-11-27_Parsed_Knowledge_Triples.csv'  # Replace with your input CSV filename
output_filename = 'files/cleaned_output.csv'  # Replace with desired output CSV filename
#utils.inputgenerator.clean_csv(input_filename, output_filename)

input_filename_ = 'files/cleaned_output.csv'  # Replace with your input CSV filename
output_filename_ = 'files/final_output.csv'  # Replace with desired output CSV filename
#utils.inputgenerator.filter_csv(input_filename_, output_filename_)


file_path = "files/final_output.csv"
vocabs = pd.read_csv(file_path)
domain = vocabs[(vocabs['Predicate'] == 'domain')] #& (vocabs['Object'] == 'Person')]
domain_cleaned = utils.inputgenerator.clean_subject_labels(domain)
domain_parents = utils.inputgenerator.transform_subject_to_hypernym(domain_cleaned)
properties_dataframe = utils.inputgenerator.generate_subject_synonyms(domain_parents) #properties
classes_dataframe = utils.inputgenerator.group_records_by_domain(properties_dataframe)

# output_f = 'output_f.csv'
# classes_dataframe.to_csv(output_f, index=False)

class_corpus = utils.inputgenerator.generate_corpus_list_c(classes_dataframe)
#class_queries = ['firs-name, address, email, affiliation, birthDate, alumniOf']
#class_queries = ['project, plan, publications, interest, workplace homepage, work info home page, school page, past project']
class_queries = ['firstname']
t_k = 10
sbert_model = 'all-MiniLM-L6-v2'

etr_output = utils.matching.etr(class_queries, class_corpus, t_k, sbert_model)

# item_to_find = 'sibling, other sector, parent, long biography, or nephew, merge company, based near, numeric datum, interests, child, death, main sector, date, event participant, residence, gender, depiction, or uncle, person participant, map, other sector, short biography, patrimony, last name, start date, logo, student, academic organization, buyer company, relationship, organization participant, document url, work role, academic participant, philantropy sector, organization participant, place, comment, main sector, spouse, ticker symbol, target company, participant, last name, political participant, documentation, source, formation year, company, philantropy sector, birth, first name, relevant date, url, grand child, tax id, cousin, grandparent, alias, connected via, representatives, legal constitution, value, end date, social reason'
found_record = utils.matching.get_record_from_item(etr_output, classes_dataframe, 'output0.csv', 'output1.csv')
#print(found_record)

