import os
import glob
import random
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os
import argparse
import random
import json
import torch
from torch import tensor 
#.... Captum imports..................
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str): # This just for initializing the pipeline
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device

    def generate_inputs(self, text: str) -> tensor: #This converts aby input text into its appropriate token input ID's
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        const_len = 7
        temp = self.__pipeline.tokenizer.encode(text, add_special_tokens=False)
        temp = temp[:const_len]
        temp += self.__pipeline.tokenizer.encode('pad',add_special_tokens=False) * max(0, const_len - len(temp))
        return torch.tensor(temp, device = self.__device)
    
    def get_tensor_from_filename(self,filename): #This function reads the concept text and coneverts them into the appropriate token input ID's
        ds = pd.read_csv(filename)
        ds = ds['Text']
        for concept in ds:
            text_indices = self.generate_inputs(concept)
            yield text_indices

    def assemble_concept(self,name, id, concepts_path):# This creates the captum.Concept objects
        dataset = CustomIterableDataset(self.get_tensor_from_filename, concepts_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=1)
        return Concept(id=id, name=name, data_iter=concept_iter)
    
    def explain(self,text): # This function is the most important one. It first computes the CAV's and then the TCAV.

        positive = self.assemble_concept('Positive',0,'positive.csv')
        neutral = self.assemble_concept('Neutral',1,'neutral.csv')
        tcav = TCAV(self.__pipeline.model, layers=['deberta.encoder.layer.11.attention.self'],save_path='./') # This line sets up the model and its layer for the TCAV procedure.
        prediction = self.__pipeline.predict(text)
        tcav.compute_cavs([[positive,neutral]]) # Computes the CAV. This line results in saving of the linear classifier.
        pos_input_texts_indices = self.generate_inputs(text)
        positive_interpretations = tcav.interpret(pos_input_texts_indices.unsqueeze(0), experimental_sets=[[positive,neutral]], target = self.__pipeline.model.config.label2id[prediction[0]['label']])# This is the bug is. Describes in the document.
        print(positive_interpretations)





def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_checkpoint = "microsoft/deberta-v3-large" 
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    clf = transformers.pipeline("text-classification", 
                            model=model, 
                            tokenizer=tokenizer, 
                            device=device
                            )
    
    exp_model = ExplainableTransformerPipeline(model_checkpoint, clf, device)
    input_text = 'This was fun!' #This is the text on which the TCAV implementation has been attempted.
    exp_model.explain(input_text)


main()
