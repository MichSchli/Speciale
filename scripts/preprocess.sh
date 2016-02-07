#!bin/bash

if [ -z "$1" ]
   then
      echo "ERROR: Input file for preprocessor not specified."
      exit
fi

INPUT_FILE=$1
FEATURE_FILE=$INPUT_FILE'.feature'
GRAPH_FILE=$INPUT_FILE'.graph'

python code/processing/ref_to_graph.py --infile $INPUT_FILE --outfile $GRAPH_FILE

python code/featurization/featurize.py --infile $INPUT_FILE --outfile $FEATURE_FILE
