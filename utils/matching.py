from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import pickle

def generate_and_save_embeddings(sentences, model_path, embeddings_file):
    # Load pre-trained Sentence BERT model
    model = SentenceTransformer(model_path)

    # Generate embeddings for sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Save embeddings to a file
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print("Embeddings saved successfully.")

#https://stackoverflow.com/questions/68334844/how-to-save-sentence-bert-output-vectors-to-a-file

def etr0(class_queries, class_corpus, t_k, sbert_model, class_embeddings):
    for query in class_queries:
        top_k = min(t_k, len(class_corpus))
        embedder = SentenceTransformer(sbert_model)
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        # class_embeddings = embedder.encode(class_corpus, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, class_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        #print("\n\n======================\n\n")
        #print("Query:", query)
        #print("\nTop 5 most similar sentences in corpus:")

        output = []
        for score, idx in zip(top_results[0], top_results[1]):
            #print(class_corpus[idx], "(Score: {:.4f})".format(score))
            #print("\n\n======================\n\n")
            output.append([class_corpus[idx], "(Score: {:.4f})".format(score)])
        return output


#https://www.appsloveworld.com/pandas/100/13/how-do-i-remove-name-and-dtype-from-pandas-output

def get_record_from_item(item_list, data, output_file_0, output_file_1):
    result_0 = []
    result_1 = []
    for sublist in item_list:
        item = sublist[0]
        score = sublist[1]

        for index, row in data.iterrows():
            corpus = row['corpus']
            corpus_items = corpus.split(', ')
            if all(keyword in corpus_items for keyword in item.split(', ')):
                new_row = row.copy()
                new_row['Score'] = score
                new_row['==='] = "============================================"
                s = new_row.to_string()
                result_0.append(s)
                result_1.append(new_row)

    result_df_0 = pd.DataFrame(result_0)
    result_df_0.to_csv(output_file_0, index=False)
    result_df_1 = pd.DataFrame(result_1)
    result_df_1.to_csv(output_file_1, index=False)
    #return result


