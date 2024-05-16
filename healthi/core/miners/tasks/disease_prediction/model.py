import time
import secrets
import bittensor as bt
from healthi.base.protocol import HealthiProtocol
from healthi.base.utils import sign_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class DiseasePredictor:
    """This class is responsible for completing disease prediction
    
    The DiseasePredictor class contains all of the code for a Miner neuron
    to generate the predicted label.

    Methods:
        execute:
            Executes the prediction within the predictor
    
    """

    def __init__(self, wallet: bt.wallet, subnet_version: int, miner_uid: int):
        # Parameters
        self.wallet = wallet
        self.miner_hotkey = self.wallet.hotkey.ss58_address
        self.subnet_version = subnet_version
        self.miner_uid = miner_uid
        # ADD MODEL HERE
        self.model_name = "Healthi/disease_prediction_v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    

    def inference(self, query, model, tokenizer):
        inputs = tokenizer.encode_plus(query, return_token_type_ids=True, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.squeeze()
        probabilities = 1 / (1 + np.exp(-logits))
        
        return probabilities.numpy() 

    def execute(self, synapse: HealthiProtocol) -> dict:
        # Responses are stored in a dict
        output = {"task": "Disease Prediction", "EHR": synapse.EHR, "predicted_probs": None}


        # Execute Model Here
        EHR = ' '.join([x for sublist in synapse.EHR for x in sublist])
        output['predicted_probs'] = self.inference(EHR, self.model, self.tokenizer).tolist()
        

        # Add subnet version and UUID to the output
        output["subnet_version"] = self.subnet_version
        output["nonce"] = secrets.token_hex(24)
        output["timestamp"] = str(int(time.time()))
        
        data_to_sign = f'{output["nonce"]}{output["timestamp"]}'

        # Generate signature for the response
        output["signature"] = sign_data(self.wallet, data_to_sign)

        synapse.output = output

        return output