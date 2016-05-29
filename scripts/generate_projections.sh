#!/bin/bash

#Configuration:
SOURCE_PARSE_FOLDER=data/bible/source_parses
NORMALIZED_PARSE_FOLDER=data/bible/normalized_parses
WALIGN_FOLDER=data/bible/projections
SALIGN_FOLDER=data/bible/projections
GOLD_FOLDER=data/universal-dependencies/test

TRAIN_FOLDER=data/experiment/train
DEV_FOLDER=data/experiment/dev
TEST_FOLDER=data/experiment/test

mkdir -p $TRAIN_FOLDER
mkdir -p $DEV_FOLDER
mkdir -p $TEST_FOLDER

TRAIN_FILE=train.conll
DEV_FILE=dev.conll
TEST_FILE=test.conll

TEMP_FILE=data/temporary/scratchpad.conll
#LANGUAGES=( "da" "de" "el" "en" "es" "fi" "fr" "he" "sl" "sv")
LANGUAGES=( "de" "es" "en" )


for LANGUAGE in ${LANGUAGES[@]}
do
    echo "Normalizing file for "$LANGUAGE$"..."
    mkdir -p $NORMALIZED_PARSE_FOLDER
    #python code/processing/proj_to_graph.py --infile $SOURCE_PARSE_FOLDER'/'$LANGUAGE'.2proj.conll' --outfile $NORMALIZED_PARSE_FOLDER'/'$LANGUAGE'.graph.conll'
done

#Do the projections:
for LANGUAGE in ${LANGUAGES[@]}
do    
    #Setup paths:
    LANGUAGE_TRAIN_FILE=$TRAIN_FOLDER'/'$LANGUAGE'-proj-'$TRAIN_FILE
    LANGUAGE_DEV_FILE=$DEV_FOLDER'/'$LANGUAGE'-proj-'$DEV_FILE
    LANGUAGE_TEST_FILE=$TEST_FOLDER'/'$LANGUAGE'-proj-'$TEST_FILE
    
    TRAIN_FEATURE_FILE=$LANGUAGE_TRAIN_FILE'.feature'
    TRAIN_GRAPH_FILE=$LANGUAGE_TRAIN_FILE'.graph'
    DEV_FEATURE_FILE=$LANGUAGE_DEV_FILE'.feature'
    DEV_GRAPH_FILE=$LANGUAGE_DEV_FILE'.graph'
    TEST_FEATURE_FILE=$LANGUAGE_TEST_FILE'.feature'
    TEST_GRAPH_FILE=$LANGUAGE_TEST_FILE'.graph'
    
    echo "Running experiment for: "$LANGUAGE
    rm -rf $TEMP_FILE

    echo "Generating empty distribution:"
    python code/multilingual/generate_zeros.py --infile $NORMALIZED_PARSE_FOLDER'/'$LANGUAGE'.graph.conll' --outfile $TEMP_FILE

    #Append information from other languages:
    for LANGUAGE2 in ${LANGUAGES[@]}
    do
	if [ $LANGUAGE != $LANGUAGE2 ]
	then
	    echo "Projecting from "$LANGUAGE2
	    python code/multilingual/project_labels.py --infile $NORMALIZED_PARSE_FOLDER'/'$LANGUAGE2'.graph.conll' --walign $WALIGN_FOLDER'/'$LANGUAGE2'-'$LANGUAGE'.bible.ibm1.reverse.wal' --salign $SALIGN_FOLDER'/'$LANGUAGE2'-'$LANGUAGE'.bible.sal' --outfile $TEMP_FILE
	fi
    done

    #Normalize:
    echo "Normalizing:"
    python code/multilingual/normalize.py --infile $TEMP_FILE
    
    #Split in train and develop:
    echo "Splitting in train and dev:"
    python code/multilingual/train_dev_split.py --infile $TEMP_FILE --train $TRAIN_GRAPH_FILE --dev $DEV_GRAPH_FILE --devsize 0.2

    #Move test:
    cat $GOLD_FOLDER'/'$LANGUAGE$'-ud-test.conllu' > $TEST_GRAPH_FILE
    
    #Featurize:
    echo "Featurizing:"
    python code/featurization/featurize.py --infile $TRAIN_GRAPH_FILE --outfile $TRAIN_FEATURE_FILE --language $LANGUAGE
    python code/featurization/featurize.py --infile $DEV_GRAPH_FILE --outfile $DEV_FEATURE_FILE --language $LANGUAGE
    python code/featurization/featurize.py --infile $TEST_GRAPH_FILE --outfile $TEST_FEATURE_FILE --language $LANGUAGE

done

    

