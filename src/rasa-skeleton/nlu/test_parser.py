import pytest
from rasa_nlu.model import Interpreter
import csv
import os

def get_test_queries():
    with open('test_queries.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [row for row in reader]
    return rows
    
def get_test_result(nlu_result):
    """
    get test elements from nlu result
    """
    result = {}
    result['entities'] = []
    result['intent'] = nlu_result['intent']['name']
    for d in nlu_result['entities']:
        result['entities'].append({'entity':d['entity'], 'value':d['value']})
        
    return result
    
    
@pytest.fixture
def nlu_parser():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(dir_path, "models", "nlu", "default", "testmodel")
    interpreter = Interpreter.load(model_dir)
    return interpreter

def test_parser(nlu_parser):
    
    test_queries = get_test_queries()
    
    for pair in test_queries:
        assert pair[1] == str(get_test_result(nlu_parser.parse(pair[0])))
    