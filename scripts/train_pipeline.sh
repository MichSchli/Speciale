#!bin/bash

if [ -z "$1" ]
   then
      echo "ERROR: Input file not specified."
      exit
fi

if [ -z "$2" ]
   then
      echo "ERROR: Algorithm not specified."
      exit
fi


INPUT_FILE=$1
FEATURE_FILE=$INPUT_FILE'.feature'
GRAPH_FILE=$INPUT_FILE'.graph'

ALGORITHM=$2

MODEL_PATH=models/$ALGORITHM'.model'

python code/parsing/train.py --features $FEATURE_FILE --sentences $GRAPH_FILE --model_path $MODEL_PATH --algorithm $ALGORITHM
