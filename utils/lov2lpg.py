import networkx as nx
import pandas as pd
import spacy
import re
from numba import jit

def create_multiple_graphs_from_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Group the DataFrame by unique values in the 'Domain' column
    domain_groups = df.groupby('Domain')

    # Create a dictionary to hold the graphs
    graphs = {}

    # Iterate through each group and create a directed graph for each domain
    for domain, group_data in domain_groups:
        G = nx.DiGraph()
        #G = nx.DiGraph()

        # Create a mapping of node labels to node IDs
        label_to_node = {}
        nodes_with_labels = []
        edges_with_labels = []

        # Create nodes with labels and populate the mapping
        for idx, row in group_data.iterrows():
            subject_label = row['Subject']
            object_label = row['Object']
            predicate_label = row['Predicate']

            if subject_label not in label_to_node:
                node_id = len(label_to_node) + 1
                label_to_node[subject_label] = node_id
                nodes_with_labels.append((node_id, {'label': subject_label}))

            if object_label not in label_to_node:
                node_id = len(label_to_node) + 1
                label_to_node[object_label] = node_id
                nodes_with_labels.append((node_id, {'label': object_label}))

            edges_with_labels.append((label_to_node[subject_label], label_to_node[object_label], {'label': predicate_label}))

        G.add_nodes_from(nodes_with_labels)
        G.add_edges_from(edges_with_labels)
        
        # Assign the graph to the corresponding domain
        graphs[domain] = G

    return graphs

def create_multiple_graphs_from_dataframe(df):
    # Group the DataFrame by unique values in the 'Domain' column
    domain_groups = df.groupby('Domain')

    # Create a dictionary to hold the graphs
    graphs = {}

    # Iterate through each group and create a directed graph for each domain
    for domain, group_data in domain_groups:
        #G = nx.Graph()
        G = nx.DiGraph()
        #G = nx.MultiDiGraph()

        # Create a mapping of node labels to node IDs
        label_to_node = {}
        nodes_with_labels = []
        edges_with_labels = []

        # Create nodes with labels and populate the mapping
        for idx, row in group_data.iterrows():
            subject_label = row['Subject']
            object_label = row['Object']
            predicate_label = row['Predicate']

            if subject_label not in label_to_node:
                node_id = len(label_to_node) + 1
                label_to_node[subject_label] = node_id
                nodes_with_labels.append((node_id, {'label': subject_label}))

            if object_label not in label_to_node:
                node_id = len(label_to_node) + 1
                label_to_node[object_label] = node_id
                nodes_with_labels.append((node_id, {'label': object_label}))

            edges_with_labels.append((label_to_node[subject_label], label_to_node[object_label], {'label': predicate_label}))

        G.add_nodes_from(nodes_with_labels)
        G.add_edges_from(edges_with_labels)
        
        # Assign the graph to the corresponding domain
        graphs[domain] = G

    return graphs

def all_connected_subgraphs_for_dict(graph_dict, m):
    def all_connected_subgraphs(g, m, name):
        def _recurse(t, possible, excluded):
            if len(t) == m:
                yield t
            else:
                for i in possible:
                    if i not in excluded:
                        new_t = (*t, i)
                        new_possible = set(g.neighbors(i)) - excluded
                        excluded.add(i)
                        yield from _recurse(new_t, new_possible, excluded)

        excluded = set()
        for node in g.nodes():
            excluded.add(node)
            yield from _recurse((node,), set(g.neighbors(node)), excluded)

    result_list = []
    for name, graph in graph_dict.items():
        connected_subgraphs = list(all_connected_subgraphs(graph, m, name))
        for subgraph in connected_subgraphs:
            result_list.append([name, nx.subgraph(graph, subgraph)])
    return result_list

from collections import defaultdict
def sum_nodes_edges(partitions):
    label_info = defaultdict(lambda: {'nodes': 0, 'edges': 0})

    for label, graph in partitions:
        label_info[label]['nodes'] += graph.number_of_nodes()
        label_info[label]['edges'] += graph.number_of_edges()

    return label_info

# label_info = sum_nodes_edges(connected_subgraphs)

def verbalize_graphs(graphs):
    verbalized_graphs = []
    
    for label,graph in graphs:
        edges = graph.edges(data=True)
        verbalized_edges = [f"{graph.nodes[src]['label']} {data['label']} {graph.nodes[dest]['label']}"
                            for src, dest, data in edges]
        
        verbalized_graph = ", ".join(verbalized_edges)
        verbalized_graphs.append([label,f"{verbalized_graph}"])
    
    return verbalized_graphs

# verbalized = verbalize_graphs(connected_subgraphs)
#print(verbalized)


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text):
    # Use regex to insert spaces between words based on capital letters
    text_with_spaces = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Apply spaCy pipeline for tokenization and lemmatization
    doc = nlp(text_with_spaces)
    
    # Extract lemmatized tokens including commas
    lemmatized_tokens = [token.text.lower() if not token.is_punct or token.text == ',' else 'comma' for token in doc]
    
    # Join the tokens to form the cleaned text
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text

def preprocess_and_merge(csv_path, descriptions):
    # Read CSV into DataFrame
    df_csv = pd.read_csv(csv_path)

    # Create DataFrame from the list of lists
    df_descriptions = pd.DataFrame(descriptions, columns=['prefix', 'Subject-Cleaned'])

    # Preprocess 'Subject-Cleaned' using spaCy
    df_descriptions['Subject-Cleaned'] = df_descriptions['Subject-Cleaned'].apply(preprocess_text)

    # Merge the two DataFrames on the 'prefix' column
    result_df = pd.merge(df_csv, df_descriptions, on='prefix')

    return result_df

from grandiso import find_motifs
def all_motif(motif, host):
    result_list = []
    for name, graph in host.items():
        matches = list(find_motifs(motif, graph))#, isomorphisms_only=True))
        #matches = list(nx.vf2pp_isomorphism(motif, graph, node_label=None))
        for match in matches:
            matched_nodes = list(match.values())
            matched_graph = graph.subgraph(matched_nodes)
            result_list.append([name, matched_graph])
    return result_list

def remove_labels_from_graph(graph):
    # Create a new graph without labels
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(graph.nodes)
    new_graph.add_edges_from(graph.edges)
    return new_graph

def partition(graph_dict, partitionValue):
    partitions = []

    def merge_helper(graph, label):
        if graph.number_of_nodes() >= partitionValue and graph.number_of_edges() > 0:
            if graph.is_directed():
                partition = [list(i) for i in nx.algorithms.community.kernighan_lin_bisection(graph.to_undirected(), partition=None, max_iter=100, weight='weight', seed=100)]
            else:
                partition = [list(i) for i in nx.algorithms.community.kernighan_lin_bisection(graph, partition=None, max_iter=100, weight='weight', seed=100)]

            lefthalf = graph.subgraph(partition[0])
            righthalf = graph.subgraph(partition[1])

            # Identify cut edges
            cut_edges = set(graph.edges()) - (set(lefthalf.edges()) | set(righthalf.edges()))

            # Check for empty partitions or unconnected graphs
            if lefthalf.number_of_edges() > 0 and righthalf.number_of_edges() > 0:
                merge_helper(lefthalf, label)
                merge_helper(righthalf, label)
            else:
                partitions.append([label, graph])
        else:
            partitions.append([label, graph])

    for label, graph in graph_dict.items():
        merge_helper(graph, label)

    return partitions



def process_subgraphs(input1, input2):
    new_subgraphs = []

    for label, g in input2:
        H = {}
        H[g] = nx.DiGraph(directed=True)
        H[g].add_nodes_from(g.nodes(data=True))
        H[g].add_edges_from(input1[label].edges(g, data=True))
        F = nx.DiGraph(directed=True)
        F.add_nodes_from((node, {'label': input1[label].nodes[node]['label']}) for node in H[g].nodes)
        F.add_edges_from(H[g].edges(data=True))
        new_subgraphs.append([label, F])

    return new_subgraphs

def connected_subgraphs(subgraphs):
    new_subgraphs = []
    for label, g in subgraphs:
        for i in nx.weakly_connected_components(g):
            connected_graph = g.subgraph(i).copy()
            new_subgraphs.append([label, connected_graph])
    
    return new_subgraphs

def filter_graphs_with_edges(input_list):
    filtered_list = [sublist for sublist in input_list if isinstance(sublist[1], nx.DiGraph) and len(sublist[1].edges) > 0]
    return filtered_list

def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list