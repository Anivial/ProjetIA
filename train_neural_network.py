# -*- coding: utf-8 -*-
'''
Train a neural network using Keras.

Organization:
    IRISA/Expression, ENSSAT
'''

import argparse
import utils
import machine_learning


#===============================================================
# Arguments and options
#===============================================================


def get_args(parser):
    '''
    Set arguments
    
    Args:
        parser: the argument parser to which arguments are added
    '''
    
    parser.add_argument("-i", "--train-input", type=str, help="File and attributes for the training set, e.g. 'train.csv:1-9,11'", required = True)
    parser.add_argument("-o", "--train-output", type=str, help="Labels for the training set, e.g. 'train.csv:12,14'", required = True)
    parser.add_argument("--train-restrict", type=str, help="Only consider these IDs in the training set, e.g. '1,2,3,5-20'")
    parser.add_argument("--train-exclude", type=str, help="Exclude these IDs from the training set, e.g. '1,2,3,5-20'")
    
    parser.add_argument("-d", "--dev-input", type=str, help="Attributes for the dev set, e.g. 'dev.csv:1-9,11'")
    parser.add_argument("-D", "--dev-output", type=str, help="Labels for the dev set, e.g. 'dev.csv:12,14'")
    parser.add_argument("--dev-restrict", type=str, help="Only consider these IDs in the dev set, e.g. '1,2,3,5-20'")
    parser.add_argument("--dev-exclude", type=str, help="Exclude these IDs from the dev set, e.g. '1,2,3,5-20'")
    
    parser.add_argument("-t", "--test-input", type=str, help="Attributes for the test set, e.g. 'test.csv:1-9,11'")
    parser.add_argument("-T", "--test-output", type=str, help="Labels for the test set, e.g. 'test.csv:12,14'")
    parser.add_argument("--test-restrict", type=str, help="Only consider these IDs in the test set, e.g. '1,2,3,5-20'")
    parser.add_argument("--test-exclude", type=str, help="Exclude these IDs from the test set, e.g. '1,2,3,5-20'")
    
    parser.add_argument("-s", "--save-model", type=str, help="File name of the output model", required = True)
    
    parser.add_argument("-v", "--verbose", help="Write in the standard output", action="store_true")
    
    parser.add_argument("-B", "--batch-size", type=int, help="Batch size")
    parser.add_argument("-L", "--loss", type=str, help="Loss function", default="mean_squared_error")
    parser.add_argument("-O", "--optimization", type=str, help="Optimization algorithm", default="sgd")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs when training the model", default=500)
    
    parser.add_argument("-r", "--recurrent", help="Whether to format data for recurrent Neural Network", dest="recurrent", action="store_true")
    parser.set_defaults(recurrent=False)
    

#===============================================================
# Main
#===============================================================

def main():
    
    global verbose

    # Parsing arguments
    
    parser = argparse.ArgumentParser(description='Train a neural network.')
    get_args(parser)
    args = parser.parse_args()

    print args

    if args.verbose:
        utils.verbose = True
        machine_learning.model_param['verbose'] = True
            
    if args.loss:
        machine_learning.model_param['loss'] = args.loss
            
    if args.optimization:
        machine_learning.model_param['optimizer'] = args.optimization
            
    if args.epochs:
        machine_learning.model_param['nb_epoch'] = args.epochs
            
    if args.batch_size:
        machine_learning.model_param['batch_size'] = args.batch_size

    
    # Training set
    
    print "######################################################################"
    print " Reading training set..."
    utils.tick()
    train_input, train_output, descriptor_labels, document_labels = utils.load_input_and_output(args.train_input,
                                                                                                args.train_output,
                                                                                                args.train_restrict,
                                                                                                args.train_exclude,
                                                                                                args.recurrent)
    if utils.verbose:
        print "-> ", len(train_input), "entries loaded"
        print "Example of input:\t", train_input[0]
        print "Example of output:\t", train_output[0]
    utils.tock()

    # Dev set
    if args.dev_input  and  args.dev_output:
        print "######################################################################"
        print " Reading dev set..."
        utils.tick()
        dev_input, dev_output, _, _ = utils.load_input_and_output(args.dev_input,
                                                                  args.dev_output,
                                                                  args.dev_restrict,
                                                                  args.dev_exclude,
                                                                  args.recurrent)
        if utils.verbose:
            print "-> ", len(dev_input), "entries loaded"
            print "Example of input:\t", dev_input[0]
            print "Example of output:\t", dev_output[0]
        utils.tock()
    
    # Test set
    if args.test_input  and  args.test_output:
        print "######################################################################"
        print " Reading test set..."
        utils.tick()
        test_input, test_output, _, _ = utils.load_input_and_output(args.test_input,
                                                                    args.test_output,
                                                                    args.test_restrict,
                                                                    args.test_exclude,
                                                                    args.recurrent)
        if utils.verbose:
            print "-> ", len(test_input), "entries loaded"
            print "Example of input:\t", test_input[0]
            print "Example of output:\t", test_output[0]
        utils.tock()
    

    print "######################################################################"
    print "Input dimension =", len(train_input[0])
    
    # Train model
    utils.tick()
    print "######################################################################"
    print "Training... "
    if args.dev_input and args.dev_output:
        model = machine_learning.train_neural_network(args.save_model,
                                                      train_input,
                                                      train_output,
                                                      dev_input,
                                                      dev_output)
    else:
        model = machine_learning.train_neural_network(args.save_model,
                                                      train_input,
                                                      train_output)
    utils.tock()
    
    if test_input and test_output:
        # Test model
        utils.tick()
        print "######################################################################"
        print "Testing... "
        machine_learning.evaluate(model,
                                  test_input,
                                  test_output,
                                  args.recurrent)
        utils.tock()

if __name__=="__main__":
    main()
