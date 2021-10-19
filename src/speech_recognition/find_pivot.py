#!/usr/bin/python

# This program organizes utterances from The Switchboard
# Dialog Act Corpus* into four parts:
# Full sentences, no-piovt sents, given and new subparts

# *http://compprag.christopherpotts.net/swda.html

import nltk
import os
from swda import CorpusReader

corpus = CorpusReader("/swda/swda/swda")

weak_verbs = ["'m", "'re", "'s", "are", "be", "did", "do",
"done", "guess", "has", "have", "'ve", "is", "mean", "seem",
"think", "thinking", "thought", "try", "was", "were"]

def add_index(utterance):
	"""Index each word-label pair in tagged sentence."""
	indexed_utterance = []
	index = 0
	for pair in utterance:
		index += 1
		indexed_utterance.append((pair, index))
	return indexed_utterance

def has_verb(indexed_utterance):
	"""Determine whether a sentence contains a verb."""
	l = []
	for i in indexed_utterance:
		if i[0][1].startswith("V"):
			l.append(i)
	if l != []:
		return True
	else:
		return False

def has_strong_verb(indexed_utterance):
	"""Determine whether a sentence contains a verb not in stop list."""
	l = []
	for i in indexed_utterance:
		if i[0][1].startswith("V") and i[0][0] not in weak_verbs:
			l.append(i)
	if l != []:
		return True
	else:
		return False

def index_strong(indexed_utterance):
	"""Determine pivot point for sentence if sentence contains a strong verb."""
	index = []
	for i in indexed_utterance:
		if i[0][1].startswith("V") and i[0][0] not in weak_verbs:
			index = i[1]
			break
	return index

def index_weak(indexed_utterance):
	"""Determine pivot point for sentence if sentence contains a weak verb."""
	index = []
	for i in indexed_utterance:
		if i[0][0] in weak_verbs:
			index = i[1]
	return index

def divide_strong(indexed_utterance):
	"""Divide sentence with strong verb into given and new parts."""
	given = []
	new = []
	for i in indexed_utterance:
		if i[1] < index_strong(indexed_utterance):
			given.append(i[0][0])
		else:
			new.append(i[0][0])
	return given, new

def divide_weak(indexed_utterance):
	"""Divide sentence with no strong verb into given and new parts."""
	given = []
	new = []
	for i in indexed_utterance:
		if i[1] <= index_weak(indexed_utterance):
			given.append(i[0][0])
		else:
			new.append(i[0][0])
	return given, new

def all_sents():
	"""Create a list of all utterances."""
	all_sents = []
	for i in corpus.iter_utterances():
		all_sents.append(i.pos_words())
	return all_sents

def all_pivotless():
	"""Create a list of all utterances with no verbs."""
	all_pivotless = []
	for i in corpus.iter_utterances():
		if not has_verb(add_index(i.pos_lemmas())):
			all_pivotless.append(i.pos_words())
	return all_pivotless


def given_and_new():
	"""Create lists of given and new utterances."""
	all_given = []
	all_new = []
	for i in corpus.iter_utterances():
		utt = add_index(i.pos_lemmas())
		if has_strong_verb(utt):
			strong_divide = divide_strong(utt)
			all_given.append(strong_divide[0])
			all_new.append(strong_divide[1])
		else:
			weak_divide = divide_weak(utt)
			all_given.append(weak_divide[0])
			all_new.append(weak_divide[1])
	return all_given, all_new

def write_sents(filename, utterances):
	"""Write sentences to file."""
	f = open(filename, "w")
	for line in utterances:
		print >> f, line

write_sents("output1.txt", all_sents())
write_sents("output2.txt", all_pivotless())
write_sents("output3.txt", given_and_new()[0])
write_sents("output4.txt", given_and_new()[1])