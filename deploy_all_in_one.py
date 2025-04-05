import os
import json
import shutil
import logging
import yaml
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_environment_file():
    """Create the environment.yml file"""
    logger.info("Creating environment.yml file...")
    
    # Define the environment configuration
    environment_config = {
        'name': 'mescyt-qa-env',
        'channels': ['conda-forge', 'defaults'],
        'dependencies': [
            'python=3.8',
            'pip=21.2.4',
            {'pip': [
                'azureml-defaults==1.48.0',
                'numpy==1.23.5',
                'pandas==1.5.3',
                'scikit-learn==1.2.2'
            ]}
        ]
    }
    
    # Write the environment configuration to a file
    with open('environment.yml', 'w') as f:
        yaml.dump(environment_config, f, default_flow_style=False)
    
    logger.info("environment.yml file created")

def create_scoring_script():
    """Create the score_retrieval.py file"""
    logger.info("Creating score_retrieval.py file...")
    
    script_content = """import os
import json
import logging
import numpy as np
from difflib import SequenceMatcher

def init():
    \"\"\"
    Initialize the QA retrieval model by loading the dataset.
    \"\"\"
    global qa_data
    global logger
    
    # Set up logging
    logger = logging.getLogger("mescyt_qa_retrieval")
    logger.setLevel(logging.INFO)
    
    # Print Python version and current directory for debugging
    import sys
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Find the model directory
    model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mescyt_qa_dataset')
    
    # If the model directory doesn't exist, try to find it
    if not os.path.exists(model_dir):
        # List all directories in the model root
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
        
        # Try to find the qa_dataset.json file
        for root, dirs, files in os.walk(model_root):
            if 'qa_dataset.json' in files:
                qa_dataset_path = os.path.join(root, 'qa_dataset.json')
                print(f"Found qa_dataset.json at: {qa_dataset_path}")
                break
        else:
            # If we didn't find it in the loop, raise an error
            raise FileNotFoundError(f"Could not find qa_dataset.json in {model_root}")
    else:
        qa_dataset_path = os.path.join(model_dir, 'qa_dataset.json')
    
    # Load the QA dataset
    print(f"Loading QA dataset from {qa_dataset_path}")
    try:
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"Successfully loaded QA dataset with {len(qa_data)} entries")
    except Exception as e:
        print(f"Error loading QA dataset: {str(e)}")
        # Try to find the file in alternative locations
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
    \"\"\"
    Calculate the similarity between two strings using SequenceMatcher.
    \"\"\"
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def run(raw_data):
    \"\"\"
    Run the QA retrieval model on the input data.
    
    Args:
        raw_data (str): JSON string containing the question.
        
    Returns:
        str: JSON string containing the answer, confidence, and matched question.
    \"\"\"
    try:
        # Parse input
        data = json.loads(raw_data)
        question = data.get('question', '')
        
        if not question:
            return json.dumps({
                'answer': 'Por favor, proporciona una pregunta.',
                'confidence': 0.0
            })
        
        # Find the most similar question in the dataset
        best_match = None
        best_score = 0.0
        
        for qa_pair in qa_data:
            score = similarity(question, qa_pair['question'])
            if score > best_score:
                best_score = score
                best_match = qa_pair
        
        # Prepare response
        if best_score >= 0.7:  # High confidence threshold
            response = {
                'answer': best_match['answer'],
                'confidence': best_score,
                'matched_question': best_match['question']
            }
        elif best_score >= 0.5:  # Medium confidence threshold
            response = {
                'answer': best_match['answer'],
                'confidence': best_score,
                'matched_question': best_match['question']
            }
        else:  # Low confidence
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
"""
    
    with open('score_retrieval.py', 'w') as f:
        f.write(script_content)
    
    logger.info("score_retrieval.py file created")

def main():
    # Create necessary files
    create_environment_file()
    create_scoring_script()
    
    # Load workspace
    logger.info("Loading workspace...")
    ws = Workspace.from_config(path="./mescyt_qa_model/config.json")
    logger.info(f"Workspace loaded: {ws.name}")
    
    # Check if service exists
    service_name = "mescyt-qa-retrieval"
    logger.info(f"Checking if service {service_name} exists...")
    try:
        service = AciWebservice(ws, service_name)
        logger.info(f"Deleting existing service {service_name}...")
        service.delete()
        logger.info("Waiting for service deletion to complete...")
        # No need to wait, the delete() method is synchronous
    except Exception as e:
        logger.info(f"Service does not exist or could not be accessed: {str(e)}")
    
    # Create a directory for the QA dataset
    os.makedirs("mescyt_qa_dataset", exist_ok=True)
    
    # Copy the QA dataset to the directory
    shutil.copy("mescyt_qa_model/qa_dataset.json", "mescyt_qa_dataset/qa_dataset.json")
    
    # Register the dataset as a model
    logger.info("Registering QA dataset as a model...")
    qa_model = Model.register(
        workspace=ws,
        model_path="mescyt_qa_dataset",
        model_name="MESCYT_QA_Dataset",
        description="MESCYT QA Dataset for retrieval-based QA"
    )
    
    # Create environment
    logger.info("Configuring environment...")
    env = Environment.from_conda_specification(
        name="mescyt-qa-env",
        file_path="environment.yml"
    )
    
    # Create inference config
    logger.info("Configuring inference...")
    inference_config = InferenceConfig(
        entry_script="score_retrieval.py",
        environment=env,
        source_directory="."
    )
    
    # Create deployment config
    logger.info("Configuring deployment...")
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=0.5,  # Reduced from 1.0 to 0.5
        memory_gb=1,    # Reduced from 2 to 1
        auth_enabled=True,
        enable_app_insights=True,
        description="MESCYT QA Retrieval-based System"
    )
    
    # Deploy the model
    logger.info("Deploying service...")
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[qa_model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True
    )
    
    # Wait for deployment to complete
    service.wait_for_deployment(show_output=True)
    
    # Print service details
    logger.info(f"Service deployed successfully!")
    logger.info(f"Service name: {service.name}")
    logger.info(f"Service state: {service.state}")
    logger.info(f"Service scoring URI: {service.scoring_uri}")
    
    # Get the primary key
    primary_key = service.get_keys()[0]
    logger.info(f"Primary key: {primary_key[:5]}...")  # Only showing first 5 chars for security
    
    # Save service details to a file
    service_details = {
        "service_name": service.name,
        "scoring_uri": service.scoring_uri,
        "primary_key": primary_key
    }
    
    with open("service_details.json", "w") as f:
        json.dump(service_details, f)
    
    logger.info("Service details saved to service_details.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        
        # Try to get logs if possible
        try:
            from azureml.core.webservice import Webservice
            service = Webservice(Workspace.from_config(path="./mescyt_qa_model/config.json"), "mescyt-qa-retrieval")
            logs = service.get_logs()
            logger.info(f"Service logs:\n{logs}")
        except Exception as log_e:
            logger.info(f"Could not retrieve logs: {str(log_e)}")
            logger.info("Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog or local deployment: https://aka.ms/debugimage#debug-locally to debug if deployment takes longer than 10 minutes.")

