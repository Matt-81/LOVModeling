import pandas as pd
import utils.scraper 
import utils.parser
import utils.inputgenerator
import utils.matching
import pandas as pd
import pickle

class_queries_input = input("Enter the query (e.g., 'firs-name, address, email, affiliation, birthDate, alumniOf'): ")
t_k_input = input("Enter the number of results (e.g., 10): ")
class_query = [class_queries_input.strip()]
t_k = int(t_k_input)

class_embeddings = pickle.load(open('class_embeddings.pkl', 'rb'))
result_dataframe = pd.read_csv('class_output.csv')
class_corpus = utils.inputgenerator.generate_corpus_list_c(result_dataframe)

sbert_model = 'all-MiniLM-L6-v2'
etr_output = utils.matching.etr0(class_query, class_corpus, t_k, sbert_model, class_embeddings)
found_record = utils.matching.get_record_from_item(etr_output, result_dataframe, 'output0.csv', 'output1.csv')


