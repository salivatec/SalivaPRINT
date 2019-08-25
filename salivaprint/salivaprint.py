import configparser
import csv
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pylab as plt
import scipy as scp
import matplotlib
import joblib
from sklearn import preprocessing
import math

VERSION = 0.1.1

# NP print options
np.set_printoptions(threshold=0.5)
np.set_printoptions(suppress=True)

def read_all_data(file):
    # READ ALL DATA -----------------------------------------------
    f = open(file)

    csv_f = csv.reader(f)
    all_data = {}

    for row in csv_f:
        # print (row)
        if row[0] in all_data:

            all_data[row[0]].append([row[1], row[2], row[3]])

        else:
            all_data[row[0]] = ([[row[1], row[2], row[3]]])

    return all_data;

def read_individual_list(file):
    aux_list = []
    f = open(file)

    csv_f = csv.reader(f)

    for row in csv_f:
        aux_list.append(row[0])

    return aux_list

def shuffle_list(list_to_shuffle):
    size = len(list_to_shuffle)
    allIDX = np.arange(size)
    np.random.shuffle(allIDX);
    aux = np.array(list_to_shuffle)
    shuffled_list = aux[allIDX];
    return shuffled_list

def check_for_presence_in_data(aux_list,all_data):
    filtered_list = []

    for ind in aux_list:
        if ind in all_data.keys():
            filtered_list.append(ind)

    return filtered_list;

def extract_features(individuals,all_data,MIN_MOL_WEIGHT,MAX_MOL_WEIGHT,NSLICES,label):

    # Creates a vector of mol weights from 7 to max_weight in order to help extracting features.
    feaures_map = np.linspace(MIN_MOL_WEIGHT, MAX_MOL_WEIGHT, num=NSLICES)

    dataset = np.zeros((len(individuals),len(feaures_map)))
    ind_ix = 0;
    for ind in individuals:
        for val in all_data[ind]:
            if float(val[0]) < MAX_MOL_WEIGHT:
                # Procura o indice em que o vector auxiliar supera o valor de mol_weight ocorrente.
                f_ix = np.where(feaures_map > float(val[0]))[0][0] # primeira ocorrencia do valor superior
                # print(val[1])
                dataset[ind_ix, f_ix-1] = float(val[1]) / float(val[2])
                # print(all_data[ind])
                # print( str(float(all_data[ind][1]) / float(all_data[ind][2])) )

        ind_ix +=1
    return dataset;

def main():
    if (len(sys.argv) < 2):
        print ('Wong number of arguments. User -h for help.')
        sys.exit()
    if (sys.argv[1] == '-v'):
        print('numpy version ' + np.__version__)
        print('scipy version ' + scp.__version__)
        print('matplotlib version ' + matplotlib.__version__)
        print('SalivaPRINT version ' + str(VERSION));
        sys.exit()

    if (sys.argv[1] == '-h'):
        print ('SalivaPRINT.\n-v - Program version. ')
        print ('-h - Program help. ')
        print ('-build outputfile - Builds dataset using config.cfg as configurations file. ')
        print ('-view inputfile - Graphical representation of the dataset. ')
        print ('-learn inputfile outputfile - Build classifier from inputfile. ')
        print ('-classify classifier_file dataset - Classify dataset. ')


    if (sys.argv[1]== '-build'):
        filename = sys.argv[2]
        config = configparser.RawConfigParser()
        config.read('config.cfg')

        MIN_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MIN_MOL_WEIGHT')
        MAX_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MAX_MOL_WEIGHT')
        NSLICES = config.getint('salivaprint_parameters', 'NSLICES')
        ALL_INDIVIDUALS_DATA_FILE = config.get('salivaprint_parameters', 'DATASET')
        HEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'CONTROL')
        UNHEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'STUDY')

        # Read data.
        all_data = read_all_data(ALL_INDIVIDUALS_DATA_FILE)

        k = all_data.keys()

        healthy = read_individual_list(HEALTHY_INDIVIDUALS_FILE)
        unhealthy = read_individual_list(UNHEALTHY_INDIVIDUALS_FILE)

        healthy = np.array(healthy)
        unhealthy = np.array(unhealthy)

        # filter list of individuals for their presence in all data
        healthy = check_for_presence_in_data(healthy, all_data)
        unhealthy = check_for_presence_in_data(unhealthy, all_data)

        l2 = healthy[:]
        l3 = unhealthy[:]

        l2.extend(l3)

        healthy.append(unhealthy)

        dataset = extract_features(l2, all_data, MIN_MOL_WEIGHT, MAX_MOL_WEIGHT, NSLICES,1)

        f = open(filename,'w')
        ix = 0
        for v in l2:
            for v2 in dataset[ix]:
                f.write(str(v2)+',')
            if(v in healthy):
                f.write(v+',0')
            else:
                f.write(v+',1')
            f.write('\n')
            ix+=1

    if (sys.argv[1] == '-view'):
        filename = sys.argv[2]
        f = open(filename,'r')
        lns = f.readlines()

        features = np.zeros((len(lns),len( lns[0].split(',')[:-2] )))
        ix = 0
        for l in lns:
            aux = l.split(',')[:-2]
            print(aux)
            features[ix] = list(map(float,aux))
            ix += 1

        plt.imshow(features,cmap='binary')
        plt.show()

    if (sys.argv[1] == '-learn'):
        config = configparser.RawConfigParser()
        config.read('config.cfg')

        MIN_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MIN_MOL_WEIGHT')
        MAX_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MAX_MOL_WEIGHT')
        NSLICES = config.getint('salivaprint_parameters', 'NSLICES')
        ALL_INDIVIDUALS_DATA_FILE = config.get('salivaprint_parameters', 'DATASET')
        HEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'CONTROL')
        UNHEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'STUDY')
        OUTPUT_FILE = sys.argv[3]


        filename = sys.argv[2]

        X = []
        y = []

        f = open(filename, 'r')
        for l in f.readlines():
            li = l.split(',')
            ind = list(map(float, li[:-2]))
            X.append(ind)
            y.append(li[-1])

        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifier.fit(X, y)

        weights = classifier.feature_log_prob_

        config = configparser.RawConfigParser()
        config.read('config.cfg')

        MIN_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MIN_MOL_WEIGHT')
        MAX_MOL_WEIGHT = config.getint('salivaprint_parameters', 'MAX_MOL_WEIGHT')
        NSLICES = config.getint('salivaprint_parameters', 'NSLICES')
        ALL_INDIVIDUALS_DATA_FILE = config.get('salivaprint_parameters', 'DATASET')
        HEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'CONTROL')
        UNHEALTHY_INDIVIDUALS_FILE = config.get('salivaprint_parameters', 'STUDY')

        v1 = np.array(weights[0])
        v2 = np.array(weights[1])

        feaures_map = np.linspace(MIN_MOL_WEIGHT, MAX_MOL_WEIGHT, num=NSLICES)
        plt.title('Influence of Molecular Weight on Classification')
        plt.xlabel('Molecular Weight')
        plt.ylabel('Influence')
        plt.plot(feaures_map,v1 - v2, 'b')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, -3, 3))

        plt.show()

        plt.figure()
        plt.plot(feaures_map,v1 - v2)
        plt.show()

        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifier.fit(X, y)

        weights = classifier.feature_log_prob_

        v1 = np.array(weights[0])
        v2 = np.array(weights[1])

        joblib.dump(classifier, OUTPUT_FILE, compress=9)

    if (sys.argv[1] == '-classify'):
        filename = sys.argv[2]
        dataset = sys.argv[3]
        clf2 = joblib.load(filename)

        data = open(dataset, 'r')

        samples = data.readlines()

        for s in samples:
            ss = s.split(',')
            ss_no_ind = ss[:-2]
            ss2 = [(list( map(float,ss_no_ind)))]
            print(ss[-2] + ' ' + str(clf2.predict_proba(ss2)[0][1])+ ' ' +str(round(clf2.predict_proba(ss2)[0][1])))

if __name__ == "__main__":
    main()