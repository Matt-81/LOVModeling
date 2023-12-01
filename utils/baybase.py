
import pandas as pd
import bnlearn as bn

file_path = "cleaned_data.csv"
vocabs = pd.read_csv(file_path)
print(vocabs)


    # Discretize automatically but with prior knowledge.
tmpdf = vocabs[['Subject', 'Predicate', 'Object']]
# Create edges
edges = [('Subject', 'Predicate'), ('Predicate', 'Object')]
# Create DAG based on edges
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, vocabs, methodtype="bayes")
print(model)

bn.plot(model, title='data set')
# bn.plot(model, interactive=True, title='method=tan and score=bic')

query = bn.inference.fit(model, variables=['Subject'], evidence={'Predicate': 'type'})
query_ = bn.inference.fit(model, variables=['Object'], evidence={'Predicate': 'domain'})

print(query_)