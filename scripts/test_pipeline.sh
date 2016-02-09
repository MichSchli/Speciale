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
OUTPUT_FILE=$INPUT_FILE'.encoded'
TEST_FILE=$INPUT_FILE'.prediction'

ALGORITHM=$2

MODEL_PATH=models/$ALGORITHM'.model'

#Predict:
python code/parsing/predict.py --features $FEATURE_FILE --sentences $GRAPH_FILE --model_path $MODEL_PATH --algorithm $ALGORITHM --outfile $OUTPUT_FILE

#Decode:
python code/decoding/decode.py --infile $OUTPUT_FILE --outfile $TEST_FILE --verbose

echo ""
echo "LAS:"

#Evaluate
python code/evaluation/score.py --gold $INPUT_FILE --prediction $TEST_FILE
