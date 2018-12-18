'''
Utilitary functions

Organization:
    IRISA/Expression.
'''

import time
import csv
import pandas as pd


verbose = False

#===============================================================
# Time
#===============================================================

global_timer = None

def tick():
    '''
    Reset and start the global timer
    '''
    global global_timer
    global_timer = time.time()


def tock():
    '''
    Stop the global timer and print the duration
    '''
    global global_timer
    global_timer = time.time() - global_timer
    print "=> TIME (s): ", global_timer


#===============================================================
# Parsing
#===============================================================

def intervals_to_set(t):
    '''
    Parse a string to array of int.
    
    Args:
    t: String to be parsed
    Return:
    List of indices
    Note:
    The string t will be splited first by E{-} and finally by E{,}. Exemple 1-20,25,100-200 => [1 2 3 .. 20 25 100 101 ... 200]
    '''
    i = 0
    result = set()
    if t:
        tiret = t.split('-')
        start = 0
        end = 0
        nb_tiret = 1
        for k in range(0,len(tiret)):
            virgul = tiret[k].split(',')
            for l in range(0,len(virgul)):
                if int(virgul[l]) not in result:
                    result.add(int(virgul[l]))
                    i += 1
                if k == 0:
                    start=int(virgul[len(virgul)-1])
                if k == nb_tiret:
                    nb_tiret += 1
                    end=int(virgul[0])
                    for j in range(start+1, end):
                        if j not in result:
                            result.add(j)
                            i += 1
                    start = int(virgul[len(virgul)-1])
    return result


def load_csv(file_name, selected_fields=[]):
    '''
    Load CSV file format using pandas.
    
    Args:
        file_name: name of the CSV file
        selected_fields: list of column IDs
    Return:
        2-uple array,array = Dataset as vectors and the first row (columns names)
    Note:
        Requires pandas.
    '''
    print file_name
    print selected_fields
    if len(selected_fields) != 0:
        data = pd.read_csv(file_name,
                           sep=',',
                           usecols=selected_fields)
    else:
         data=pd.read_csv(file_name,
                          sep=',')
    labels = pd.read_csv(file_name,
                         header=None,
                         nrows=1,
                         sep=','
                         ).as_matrix()[0]
    return data.as_matrix(), labels
 



def load_input_and_output(input_arg, output_arg, restricted_ids=None, excluded_ids=None, recurrent=False):
    '''
    Load input attributes and associated outputs from CSV files
    First line is supposed to contain field labels
    
    Args:
        input_arg: string formatted as "file_name:fields_ids"
        output_arg: string formatted as "file_name:fields_ids"
        restricted_ids: list of fields IDs for the descriptors, default is all is 
        excluded_ids: list of fields IDs to be excluded for the descriptors
    Return:
        4 arrays: input_vectors, output_vectors, labels of the input attributes, labels of the output values
    '''
    
    global args
    
    input_vectors = []
    output_vectors = []
    
    input_file, input_intervals = input_arg.split(":")
    output_file, output_intervals = output_arg.split(":")
    input_fields = intervals_to_set(input_intervals)
    output_fields = intervals_to_set(output_intervals)
    
    tmp_input, input_labels = load_csv(input_file, input_fields)
    tmp_output, output_labels = load_csv(output_file, output_fields)
    
    if len(tmp_input) != len(tmp_output):
        raise Exception("Lengths of ", input_file, " and ", output_file, " differ.")
    else:
        ids = set(range(len(tmp_input)))
        
    if recurrent:
        tmp_input = tmp_input.reshape((tmp_input.shape[0], tmp_input.shape[1], 1))

    if restricted_ids:
        ids &= intervals_to_set(restricted_ids)
    
    if excluded_ids:
        ids -= intervals_to_set(excluded_ids)
    
    for i in ids:
        #if verbose:
            #print "Add entry", i
        input_vectors.append(tmp_input[i])
        output_vectors.append(tmp_output[i])
        
    return input_vectors, output_vectors, input_labels, output_labels


