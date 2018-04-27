"""Example of training spaCy's named entity recognizer, starting off with an existing model or a blank model.
For more details, see the documentation: * Training: https://spacy.io/usage/training 
										 * NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+"""

from __future__ import unicode_literals, print_function

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

argumentParser = argparse.ArgumentParser(description='trains the NER model with custom defined tags', epilog='Mail bug reports and suggestions to <iscope-bugreports@innominds.com>')
argumentParser.add_argument('--local_file_path', dest='local_file_path', help='path to project dir')
argumentParser.add_argument('--local_dest_path', dest='local_dest_path',help = 'path to store the Named Entities')
argumentParser.add_argument('--header', dest = 'header', help = 'this is know if the header is present in the file provided')

args = argumentParser.parse_args()

LOCAL_FILE_PATH = args.local_file_path
LOCAL_DEST_PATH = args.local_dest_path
HEADER 			= args.header
#NER_INPUT_PATH 	= args.NER_input_path

#Generating the log file

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train_ner_log.log',
                    filemode='w')

                            #================================================#
                            #       NER EXTRACTION FROM THE TEXT DATA        #
                            #================================================#

if not (LOCAL_FILE_PATH and LOCAL_DEST_PATH):
	msg = "Please provide parameters for LOCAL FILE PATH TO TRAIN THE NER MODEL.\n"
	print(msg)
	logger.debug(msg)
	parser.print_help()
	exit(1)

##############################################################################################################################
#											METHOD TO READ THE FILE
##############################################################################################################################

#Redaing the csv file from the path specified
def read_file(FILENAME, filetype, header):
    if header is None:
        if filetype == 'csv':
			ner_df = pd.read_csv(FILENAME, names=['ner_data'])
        elif (filetype == 'xlsx') | (filetype == 'xls'):
			ner_df = pd.read_excel(FILENAME, names=['ner_data'])
        elif filetype == 'txt':
            ner_df = pd.read_table(FILENAME, names = ['ner_data'])
	
    elif header is not None:
        if filetype == 'csv':
			ner_df = pd.read_csv(FILENAME, header = None, names = ['ner_data'])
        elif (filetype == 'xls') | (filetype == 'xlsx'):
			ner_df = pd.read_excel(FILENAME, header = None, names = ['ner_data'])
        elif filetype == 'txt':
            ner_df = pd.read_table(FILENAME, header = None, names = ['ner_data'])
    return ner_df

##############################################################################################################################
#											READING THE FILE PROVIDED
##############################################################################################################################
if LOCAL_FILE_PATH is not None:

    #Extract the extension of the filename
    filetype = LOCAL_FILE_PATH.split('.')  #splitting the file name by '.' to get the file type
    file_ext = filetype[-1]

    try:
		ner_df = read_file(LOCAL_FILE_PATH, file_ext, HEADER)
		ner_df.columns = ['ner_data']
    except:
		msg = "Please provide Review file with extensions as .csv, .txt or .json"
		print(msg)
		logger.debug(msg)
		exit(1)

##############################################################################################################################
#                                           TEST THE TRAINED MODEL
##############################################################################################################################
#Load the text data into the variable
data = ner_df['ner_data']

#Read the test data from which you can read the test reviews and extract the entities from it
model_path = os.path.join(os.getcwd(),'model')

nlp2 = spacy.load(model_path) #load the spacy trained model
ent_dict = {}
for text in data:
    doc2 = nlp2(text.decode('utf8'))
    for ent in doc2.ents:
        ent_dict[ent.text] = ent.label_
        print(ent.label_, ent.text)

#Load the dictionary in the LOCAL DEST PATH
import json
with open(LOCAL_DEST_PATH, 'w') as file:
    file.write(json.dumps(ent_dict))