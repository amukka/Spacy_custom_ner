"""Example of training spaCy's named entity recognizer, starting off with an existing model or a blank model.
For more details, see the documentation: * Training: https://spacy.io/usage/training 
										 * NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+"""

#from __future__ import unicode_literals, print_function

#Importing the necessary packages
import plac
import random
from pathlib import Path
import spacy
import os
import re
import pandas as pd
import numpy as np
import commands
import logging
import nltk
import argparse

argumentParser = argparse.ArgumentParser(description='trains the NER model with custom defined taggs', epilog='Mail bug reports and suggestions to <iscope-bugreports@innominds.com>')
argumentParser.add_argument('--local_file_path', dest='local_file_path', type=str, help = 'this is the type of string on which the custom NERs are provided as output')
argumentParser.add_argument('--header', dest = 'header', help = 'this is know if the header is present in the file provided')
#argumentParser.add_argument('--ner_dest_path', dest='ner_dest_path',help = 'path to store the extracted Named Entities from the trained model')

args = argumentParser.parse_args()

LOCAL_FILE_PATH = args.local_file_path
HEADER 			= args.header

#Generating the log file

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='ner_model_train.log',
                    filemode='w')

                            #=====================================================#
                            # MODEL BUILDING FOR CUSTOM NAMED ENTITY RECOGNITION  #
                            #=====================================================#

if not (LOCAL_FILE_PATH):
	msg = "Please provide parameters for LOCAL FILE PATH TO TRAIN THE NER MODEL.\n"
	print(msg)
	logger.debug(msg)
	# parser.print_help()
	exit(1)


##############################################################################################################################
#											METHOD TO READ THE FILE
##############################################################################################################################

#Redaing the csv file from the path specified
def read_file(FILENAME, filetype, header):
    if header is None:
		if filetype == 'csv':
			ner_df = pd.read_csv(FILENAME, names=['text','features','entity'])

		elif (filetype == 'xlsx') | (filetype == 'xls'):
			ner_df = pd.read_excel(FILENAME, names=['text','features','entity'])
	
    elif header is not None:
        if filetype == 'csv':
            ner_df = pd.read_csv(FILENAME, header = None, names = ['text','features','entity'])
        
        elif (filetype == 'xls') | (filetype == 'xlsx'):
            ner_df = pd.read_excel(FILENAME, header = None, names = ['text','features','entity'])
    return ner_df

# Definition to make the data as tuples
# def get_train_data_tuple(text, features, entity='APP'):
#     if not text:
#         return None
#     result = {'entities': []}
#     i = 0
#     for feature in features:
#         token = feature
#         token = re.escape(token)
#         if len(entity) == 1:
#             for m in re.finditer(token, text, re.IGNORECASE):
#                 entity = ''.join(e for e in entity)
#                 result['entities'].append(((m.start()+1), m.end(), entity))
#         else:
#             for m in re.finditer(token, text, re.IGNORECASE):
#                 if i == 0:
#                     result['entities'].append((m.start(), m.end(), entity[0]))
#                     i += 1
#                 else:
#                     result['entities'].append(((m.start()+1), m.end(), entity[i]))
#                     i += 1
#     return text, result    

def get_train_data_tuple(text, features, entity='APP'):
    if not text:
        return None
    result = {'entities': []}
    i = 0
    for feature in features:
        token = feature
        token = re.escape(token)
        for m in re.finditer(token, text, re.IGNORECASE):
            if len(entity) == 1:
                ent = ''.join([e for e in entity])
                result['entities'].append(((m.start()), m.end(), ent))
            else:
                if i == 0:
                    result['entities'].append((m.start(), m.end(), entity[0]))
                    i += 1
                else:
                    result['entities'].append(((m.start()+1), m.end(), entity[i]))
                    i += 1
    return text, result

##############################################################################################################################
#											READING THE FILE PROVIDED
##############################################################################################################################

if LOCAL_FILE_PATH is not None :

    filetype = LOCAL_FILE_PATH.split('.')
    file_ext = filetype[-1]	
    try:
        ner_df = read_file(LOCAL_FILE_PATH, file_ext, HEADER)
        ner_df.columns = ['text','features','entity']

    except Exception as e: print(e)
		# msg = "Please provide Review file with extensions as .csv, .txt or .json"
		# print(msg)
		# logger.debug(msg)
		
##############################################################################################################################
#											PREPARATION FOR THE TRAIN DATA
##############################################################################################################################

data = ner_df   #Assigning the dataframe to data variable

# Each row of the dataframe is sent to the function and it is returning as text with entity. And this is appended to the TRAIN_DATA list
TRAIN_DATA = []
try:
    for index, row in data.iterrows():
        feature = list(row[1].split(','))
        ent     = list(row[2].split(','))
        TRAIN_DATA.append(get_train_data_tuple(row[0], feature, ent))
except:
    msg = "There is some issue in train data preparation. Have a look at the method 'get_train_data_tuple'"
    print(msg)
    logger.debug(msg)
    exit(1)

print(TRAIN_DATA)
##############################################################################################################################
#											MODEL TO TRAIN THE CUSTOM NER ENTITIES
##############################################################################################################################

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=20):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(str(ent[2]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
            
    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        
    # save model to output directory
    if output_dir is not None:
        print(output_dir)
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

##############################################################################################################################
#                                           CALLING THE MODEL
##############################################################################################################################

# Defining the model path
model_path = os.path.join(os.getcwd(),'model')

#calling the main function so that it gets trained and stores the output to the model_path directory
try:
    main(output_dir = model_path)
except:
    msg = "Training the model is not done properly and some exception has occured."
    print(msg)
    logger.debug(msg)
    exit(1)


#Information on the trained model
msg = 'The data is trained properly and is loaded at the path '+ model_path
print(msg)
logger.info(msg)