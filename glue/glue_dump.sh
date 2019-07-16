#!/bin/bash
GLUE_DATA=${GLUE_DATA:="./glue_data"}
GLUE_TASK=${GLUE_TASK:="CoLA"}

case ${GLUE_TASK} in
# Single Sentence Tasks
    
    # Corpus of Linguistic Acceptability
    # Single sentences annotated whether they are gramatical sentences
    CoLA)
	validation_file=${GLUE_DATA}/CoLA/dev.tsv
	awk_fields='{ print "label="$2 "\tsentence="$4 }'
	;;
    # Stanford Sentiment Treebank
    # Single sentences from movie reviews. Annotated whether they are positive or negative.
    SST)
	validation_file=${GLUE_DATA}/SST-2/dev.tsv
	awk_fields='{ print "label="$2 "\tsentence="$1 }'
	;;

# Similarity and paraphrase tasks    
    # Microsoft Research Paraphrase Corpus
    # Pairs of sentences from online news sources, are they semantically equivalent?
    # Unbalanced (68% positive)
    MRPC)
	validation_file=${GLUE_DATA}/MRPC/dev.tsv
	awk_fields='{ print "label="$1 "\n\tsentence="$4 "\n\tsentence2="$5 }'
	;;

    # Quora Question Pairs
    # Are the questions semantically equivalent?
    # Unbalanced (63% negative)
    QQP)
	validation_file=${GLUE_DATA}/QQP/dev.tsv
	awk_fields='{ print "label="$6 "\n\tsentence="$4 "\n\tsentence2="$5 }'
	;;

    # Semantic Textual Similarity
    # Sentence pairs drawn from news headlines, are they similar? (0 to 5)
    STS)
	validation_file=${GLUE_DATA}/STS-B/dev.tsv	
	awk_fields='{ print "label="$10 "\n\tsentence="$8 "\n\tsentence2="$9 }'
	;;

# Inference tasks    
    # Multi-Genre Natural Language Inference Corpus
    # Given a premise + hypothesis; does the premise entail the hypothesis, contradict the hypothesis or is it neutral
    # Looks at both matched (in domain) and mis-matched (cross-domain)
    MNLI)
	validation_file=${GLUE_DATA}/MNLI/dev_matched.tsv
	awk_fields='{ print "label="$16 "\n\tsentence="$9 "\n\tsentence2="$10 }'
	;;

    # Stanford Question Answering Dataset
    # Does the second sentence answer the question?
    QNLI)
	validation_file=${GLUE_DATA}/QNLI/dev.tsv
	awk_fields='{ print "label="$4 "\n\tquestion="$2 "\n\tsentence2="$3 }'
	;;

    # Recognizing Textual Entailment
    # Does the second sentence agree with the first
    RTE)
	validation_file=${GLUE_DATA}/RTE/dev.tsv
	awk_fields='{ print "label="$4 "\n\tsentence="$2 "\n\tsentence2="$3 }'
	;;

    # Winograd Schema Challenge
    # Does the second sentence have the proper pronoun replacement
    WNLI)
	validation_file=${GLUE_DATA}/WNLI/dev.tsv
	awk_fields='{ print "label="$4 "\n\tsentence="$2 "\n\tsentence2="$3 }'
	;;    

    
    *)
	echo "Unknown GLUE_TASK: ${GLUE_TASK}"
	exit 1
esac

cat $validation_file | awk -F '\t' "$awk_fields"

