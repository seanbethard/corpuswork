import nltk

grammar = nltk.data.load('file:agree_join.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar)

agreement_test_sentences = ['Mary sang and danced',
                            'Mary and John danced',
                            'Mary and John',
                            'Mary and John saw Kim',
                            'Mary saw John and Kim',
                            'Mary sang and John danced',
                            'Mary baked a soft and tasty cake',
                            'Mary baked a tasty quiche and a soft cake']

for sent in agreement_test_sentences:
    print sent + '\n'
    trees = parser.nbest_parse(sent.split())
    if len(trees) == 0:
        print '--> ungrammatical\n'
    else:
        for tree in trees:
            print tree
            print '\n'
