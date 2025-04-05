import os
import json
import logging
import numpy as np
from difflib import SequenceMatcher

def init():
    global qa_data
    global logger
    
    logger = logging.getLogger("mescyt_qa_retrieval")
    logger.setLevel(logging.INFO)
    
    import sys
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mescyt_qa_dataset')
    
    if not os.path.exists(model_dir):
        model_root = os.getenv('AZUREML_MODEL_DIR')
        print(f"Model root directory: {model_root}")
        print("Contents of model root directory:")
        for item in os.listdir(model_root):
            print(f"  {item}")
            item_path = os.path.join(model_root, item)
            if os.path.isdir(item_path):
                print(f"    Contents of {item}:")
                for subitem in os.listdir(item_path):
                    print(f"      {subitem}")
        
        for root, dirs, files in os.walk(model_root):
            if 'qa_dataset.json' in files:
                qa_dataset_path = os.path.join(root, 'qa_dataset.json')
                print(f"Found qa_dataset.json at: {qa_dataset_path}")
                break
        else:
            raise FileNotFoundError(f"Could not find qa_dataset.json in {model_root}")
    else:
        qa_dataset_path = os.path.join(model_dir, 'qa_dataset.json')
    
    print(f"Loading QA dataset from {qa_dataset_path}")
    try:
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"Successfully loaded QA dataset with {len(qa_data)} entries")
    except Exception as e:
        print(f"Error loading QA dataset: {str(e)}")
        alternative_paths = [
            '/var/azureml-app/qa_dataset.json',
            os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'qa_dataset.json'),
            os.path.join(os.getenv('AZUREML_MODEL_DIR'), '2', 'mescyt_qa_dataset', 'qa_dataset.json')
        ]
        
        for path in alternative_paths:
            print(f"Trying alternative path: {path}")
            if os.path.exists(path):
                print(f"Found file at: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                print(f"Successfully loaded QA dataset with {len(qa_data)} entries")
                break
        else:
            raise FileNotFoundError(f"Could not find qa_dataset.json in any expected location")

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def run(raw_data):
    try:
        data = json.loads(raw_data)
        question = data.get('question', '')
        
        if not question:
            return json.dumps({
                'answer': 'Por favor, proporciona una pregunta.',
                'confidence': 0.0
            })
        
        best_match = None
        best_score = 0.0
        
        for qa_pair in qa_data:
            score = similarity(question, qa_pair['question'])
            if score > best_score:
                best_score = score
                best_match = qa_pair
        
        if best_score >= 0.7:
            response = {
                'answer': best_match['answer'],
                'confidence': best_score,
                'matched_question': best_match['question']
            }
        elif best_score >= 0.5:
            response = {
                'answer': best_match['answer'],
                'confidence': best_score,
                'matched_question': best_match['question']
            }
        else:
            response = {
                'answer': 'Lo siento, no tengo suficiente información para responder a esa pregunta específica.',
                'confidence': best_score
            }
            if best_match:
                response['matched_question'] = best_match['question']
        
        return json.dumps(response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error in run method: {error_msg}")
        return json.dumps({
            'error': error_msg,
            'answer': 'Lo siento, ocurrió un error al procesar tu pregunta.'
        })