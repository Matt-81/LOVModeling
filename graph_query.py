import pickle
import utils.matching
import pandas as pd
import utils.inputgenerator

query_input = input("Enter the query (e.g., 'dance group sub class of performing group, dance group sub class of performing group'): ")
t_k_input = input("Enter the number of results (e.g., 10): ")
query = [query_input]
t_k = int(t_k_input)

#####

graph_embeddings = pickle.load(open('embeddings.pkl', 'rb'))
result_dataframe = pd.read_csv('verbalized.csv')
graph_corpus = utils.inputgenerator.generate_corpus_list_c(result_dataframe)

sbert_model = 'all-MiniLM-L6-v2'
etr_output = utils.matching.etr0(query, graph_corpus, t_k, sbert_model, graph_embeddings)
found_record = utils.matching.get_record_from_item(etr_output, result_dataframe, 'output_0.csv', 'output_1.csv')

