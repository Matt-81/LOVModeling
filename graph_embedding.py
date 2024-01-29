import utils.lov2lpg
import utils.inputgenerator
import utils.matching
import pandas as pd
#import networkx as nx
import pickle

file_path = "files/final_output.csv"

predicate_values_to_keep = ['subClassOf']#, 'isDefinedBy']
#predicate_values_to_keep = [query.strip() for query in predicate_values_to_keep_input.split(',')]
domain_values_to_keep = ['vivo', 'schema']#, 'vivo']#, 'isDefinedBy']
#domain_values_to_keep = [query.strip() for query in domain_values_to_keep_input.split(',')]
object_values_to_keep = ['Place']#, 'isDefinedBy']
query = ['dance group sub class of performing group']
#query = [query_input.strip()]
t_k = 200
#t_k= int(t_k_input)

####

filtered_predicates = utils.inputgenerator.filter_dataframe_by_predicates(pd.read_csv(file_path), predicate_values_to_keep)
filtered_domain_and_predicates = utils.inputgenerator.filter_dataframe_by_domain(filtered_predicates, domain_values_to_keep)
# # Call the function and get the graphs
result_graphs = utils.lov2lpg.create_multiple_graphs_from_dataframe(filtered_domain_and_predicates) #or result1
#graphs = [graph for domain, graph in result_graphs.items()]

###partition0
subgraphs = utils.lov2lpg.partition(result_graphs, 10) #int(partitions_input))
subgraphs_recovered_edges = utils.lov2lpg.process_subgraphs(result_graphs, subgraphs)
subgraphs_connected = utils.lov2lpg.connected_subgraphs(subgraphs_recovered_edges)
subgraphs_connected_with_edges = utils.lov2lpg.filter_graphs_with_edges(subgraphs_connected)
#sum = utils.lov2lpg.sum_nodes_edges(subgraphs_connected_with_edges)
###partition1
subgraphs_0 = utils.lov2lpg.partition(result_graphs, 20) #int(partitions_input))
subgraphs_recovered_edges_0 = utils.lov2lpg.process_subgraphs(result_graphs, subgraphs_0)
subgraphs_connected_0 = utils.lov2lpg.connected_subgraphs(subgraphs_recovered_edges_0)
subgraphs_connected_with_edges_0 = utils.lov2lpg.filter_graphs_with_edges(subgraphs_connected_0)
#sum = utils.lov2lpg.sum_nodes_edges(subgraphs_connected_with_edges_0)

###merging
merged = utils.lov2lpg.Union(subgraphs_connected_with_edges,subgraphs_connected_with_edges_0)
#subgraphs_viz = utils.inputgenerator.visualize_networkx_graphs(merged)

###verbalization
verbalized = utils.lov2lpg.verbalize_graphs(merged)
csv_path = 'files/LOV_vocabs.csv'
result_dataframe = utils.lov2lpg.preprocess_and_merge(csv_path, verbalized) #this is slow
result_dataframe.to_csv('verbalized.csv', index=False)

###embedding
# Example sentences
class_corpus = utils.inputgenerator.generate_corpus_list_c(result_dataframe)
# Path to pre-trained Sentence BERT model
model_path = 'paraphrase-MiniLM-L6-v2'
# File to save embeddings
embeddings_file = 'embeddings.pkl'
# Save
save_pkl = utils.matching.generate_and_save_embeddings(class_corpus,model_path,embeddings_file)







