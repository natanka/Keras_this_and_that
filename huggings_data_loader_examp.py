from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers.data.processors.utils import InputFeatures
import numpy as np
from torch.utils.data import DataLoader

class sturcc_to_holder:
    def __init__(self, list_files):
        self.list_of_file = list_files

    #with tensors
    def __getitem__(self, idx):
        aa = torch.load(self.list_of_files[idx])
        return aa[0], aa[1], aa[2]

    def __len__(self):
        return len(self.list_of_files)

    #without toensors
    def __getitemnumpy__(self, idx):
        aa = np.load(self.list_of_files[idx], allow_pickle=True)
        cc = aa.data.obj.tolist()
        c1 = cc.input_ids
        c2 = cc.attention_mask
        c3 = cc.label
        return torch.tensor(c1), torch.tensor(c2), c3


def create_toy_dataset():
    dataset00 = load_dataset('glue', 'mrpc', split='train')
    ldata= len(dataset00)
    xx = np.random.randint(3 , size=ldata)
    documents = [dataset00[i]["sentence1"] + " " + dataset00[i]["sentence2"] for i in range(ldata)]
    return xx, documents


def regular_procedure(tokenizer,documents , labels ):
    tokens = tokenizer.batch_encode_plus(documents )

    features=[InputFeatures(label=labels[j], **{key: tokens[key][j] for key in tokens.keys()}) for j in
                range(len(documents))]
    return features

def generate_files_no_tensor(tokenizer,documents, labels ):
    tokens  = tokenizer.batch_encode_plus(documents )


    file_pref ="my_file_"
    for j in range(len(documents) ):
            inputs = {k: tokens[k][j] for k in tokens}
            feature = InputFeatures(label=labels[j], **inputs)
            file_name = file_pref +"_"+str(j)+".npy"
            np.save(file_name, np.array(feature))
    return



def generate_files_with_tensor(tokenizer,documents, labels ):
    tokens = tokenizer.batch_encode_plus(doc0, return_tensors='pt')

    file_pref ="my_file_"
    for j in range(len(documents) ):
        file_name = file_pref +"_"+str(j)+".pt"
        input_t =  torch.squeeze(torch.index_select(tokens["input_ids"],dim=0,index=torch.tensor(j)))
        input_m =   torch.squeeze(torch.index_select(tokens["attention_mask"],dim=0,index=torch.tensor(j)))
        torch.save([input_t,input_m,labels[j]], file_name)
    return

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    lab0, doc0 =create_toy_dataset()
    feature =regular_procedure(tokenizer, doc0, lab0  )

    #Preparing tensors
    generate_files_with_tensor(tokenizer, doc0, lab0)

    #Numpy files -little slower in training
    generate_files_no_tensor( tokenizer, doc0, lab0)


    #Preparing dataloader
    list_of_files= []
    your_data = sturcc_to_holder(list_files=list_of_files)

    dss = DataLoader(your_data, batch_size=16, shuffle=True)
