import nltk

grammar = nltk.data.load('file:aux.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


agreement_test_sentences = ['John does eat apples',
							'John does have been eating apples']


for sent in agreement_test_sentences:
    print sent + '\n'
    trees = parser.nbest_parse(sent.split())
    if len(trees) == 0:
        print '--> ungrammatical\n'
    else:
        for tree in trees:
			print tree
			print '\n'
			tree.draw()