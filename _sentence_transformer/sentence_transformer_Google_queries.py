import numpy as np
import torch
import boto3
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tarfile
import io

class TopHeaders():
    def __init__(
        self,
        model_path,
        s3_bucket,
        file_prefix
        ):


        self.s3_client = boto3.client(
            's3'
            )

        self.model_path = model_path
        self.s3_bucket = s3_bucket
        self.file_prefix = file_prefix
        self.model, self.tokenizer = self.from_pretrained()

    def from_pretrained(self):
        model = self.load_model_from_s3()
        tokenizer = self.load_tokenizer()
        return model, tokenizer

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return tokenizer

    def load_model_from_s3(self):
        if self.model_path and self.s3_bucket and self.file_prefix:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{self.model_path}/config.json')
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    state = torch.load(io.BytesIO(f.read()))
                    model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=None,
                    state_dict=state,
                    config=config
                    )
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def remove_edit_tag(self, headers):
        """
        In wikipedia headers, [edit] tag is common. Remove it.
        """
        return [(h.replace("[edit]", "")).strip() for h in headers]


    ####################### From HuggingFace Hub (below) ####################### 

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        print('model_output extracting 0 indx elm')
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    def compute_embeddings(self, input_):
        #Tokenize input
        print('encoding this input using tokenizer...')
        encoded_input = self.tokenizer(input_, input_, padding=True, truncation=True, return_tensors='pt')
        print('computing emd using model')
        #Compute query embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        print('max pooling this input.')
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    ####################### From HuggingFace Hub (above) ####################### 

    def __call__(
        self,
        keyword_searched,
        raw_headers,
        estimated_frac_of_top_headers=3
        ): #get_top_headers
        print('computing embeddings., ')
        keyword_searched_embd = self.compute_embeddings(keyword_searched)
        headers_emdb = self.compute_embeddings(raw_headers)
        print('computing cosine simi., ')
        cos_scores = self.pytorch_cos_sim(keyword_searched_embd, headers_emdb)[0]
        # Sort the results in decreasing order and get the first top_k
        top_k = len(raw_headers) // estimated_frac_of_top_headers
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        refined_headers = []
        for idx in top_results[0:top_k]:
            refined_headers.append(raw_headers[idx])
        
        top_headers_str = ",".join(list(set(refined_headers)))
        return top_headers_str



    ####################### Taken from Sentence Transformer (to avoid dependency) (below) ############# 

    def pytorch_cos_sim(self, a, b):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        return self.cos_sim(a, b)

    def cos_sim(self, a, b):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    ####################### Taken from Sentence Transformer (to avoid dependency) (above) ############# 
