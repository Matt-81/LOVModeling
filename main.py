import pandas as pd
import utils.scraper 
import utils.parser
import utils.inputgenerator

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

####  CHAT WITH LOV
file_path = "files/final_output.csv"
vocabs = pd.read_csv(file_path)
filtered_vocabs = vocabs[(vocabs['Predicate'] == 'domain') & (vocabs['Object'] == 'Person')]
#filtered_df = df[((df['predicate'] == 'domain') | (df['predicate'] == 'range')) & (df['object'] == 'Person')]
#print(filtered_vocabs)
filtered_vocabs_sop = filtered_vocabs[['Subject', 'Predicate', 'Object']]
#print(filtered_vocabs_sop)
# edges = [('Object', 'Subject'), ('Predicate', 'Subject')]
edges = [('Object', 'Subject')] #0.7 :: Person :- Name 
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, filtered_vocabs_sop, methodtype="bayes")
bn.plot(model, title='LOV')
#CPDs = bn.print_CPD(DAG)
q1 = bn.inference.fit(model, variables=['Subject'], evidence={'Object': 'Person'}) #evidence={'Predicate':'domain', 'Object': 'Address'})
print(q1)

#check top-level ontology for ontology alignment

#example query 1
#given multiple properties, probabilities of associated classes

# # USE CASE 1 #https://erdogant.github.io/bnlearn/pages/html/UseCases.html
# # Load dataframe
# df = bn.import_example()
# # Import DAG
# DAG = bn.import_DAG('sprinkler', CPD=False)
# # Learn parameters
# model = bn.parameter_learning.fit(DAG, df)
# # adjacency matrix:
# model['adjmat']
# edges = [('Cloudy', 'Sprinkler'),
#          ('Cloudy', 'Rain'),
#          ('Sprinkler', 'Wet_Grass'),
#          ('Rain', 'Wet_Grass')]
# df = bn.import_example('sprinkler')
# import bnlearn as bn
# DAG = bn.make_DAG(edges)
# # [BNLEARN] Bayesian DAG created.
# # Print the CPDs
# CPDs = bn.print_CPD(DAG)
# # [BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.
# #bn.plot(DAG)
# DAG = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
# CPDs = bn.print_CPD(DAG)
# q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
# #print(q1.df)

