import pandas as pd
import utils.scraper 
import utils.parser
import utils.inputgenerator
import utils.matching
import pandas as pd

file_path = "files/final_output.csv"
vocabs = pd.read_csv(file_path)
domain = vocabs[(vocabs['Predicate'] == 'domain')] #& (vocabs['Object'] == 'Person')]
domain_cleaned = utils.inputgenerator.clean_subject_labels(domain)
domain_parents = utils.inputgenerator.transform_subject_to_hypernym(domain_cleaned)
properties_dataframe = utils.inputgenerator.generate_subject_synonyms(domain_parents) #properties
#classes_dataframe = utils.inputgenerator.group_records_by_domain(properties_dataframe)

classes_dataframe = utils.inputgenerator.group_records_by_domain(properties_dataframe)
class_output = 'class_output.csv'
classes_dataframe.to_csv(class_output, index=False)

class_corpus = utils.inputgenerator.generate_corpus_list_c(classes_dataframe)

# Path to pre-trained Sentence BERT model
model_path = 'paraphrase-MiniLM-L6-v2'
# File to save embeddings
embeddings_file = 'class_embeddings.pkl'
# Save
save_pkl = utils.matching.generate_and_save_embeddings(class_corpus,model_path,embeddings_file)
