import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F



class SentenceTransformer(nn.Module):
    def __init__(self,tokenizer_name='sentence-transformers/all-mpnet-base-v2',model_name='sentence-transformers/all-mpnet-base-v2',device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def forward(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self,sentences):
        tokens=self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens


class PoMP(nn.Module):
    def __init__(self,in_dim,attention_dim,gender_dim,pregnancy_dim,num_heads,dropout,category_num,disease_num,tokenizer_name='sentence-transformers/all-mpnet-base-v2',model_name='sentence-transformers/all-mpnet-base-v2',device='cpu'):
        super().__init__()
        self.device = device

        self.sentence_transformer=SentenceTransformer(tokenizer_name,model_name,device)
        self.sentence_dim=self.sentence_transformer.model.config.hidden_size

        self.gender_emb=nn.Embedding(2,gender_dim).to(self.device)
        self.pregnancy_emb = nn.Embedding(2, pregnancy_dim).to(self.device)
        self.linear1=nn.Linear(gender_dim+pregnancy_dim+in_dim,attention_dim,dtype=torch.float64)
        self.attention1=nn.MultiheadAttention(embed_dim=attention_dim,num_heads=num_heads,dropout=dropout,dtype=torch.float64,batch_first=True)
        self.layer_norm=nn.LayerNorm(attention_dim,dtype=torch.float64)

        self.category_num = category_num
        self.disease_num= disease_num
        self.disease_max_num=max([d for d in self.disease_num])
        self.attention2=nn.MultiheadAttention(embed_dim=attention_dim+self.sentence_dim,num_heads=num_heads,dropout=dropout,dtype=torch.float64,batch_first=True)


        self.category_linear=nn.Linear(attention_dim+self.sentence_dim,category_num,dtype=torch.float64)
        self.category_norm = nn.LayerNorm(category_num, dtype=torch.float64)
        self.softmax = nn.Softmax()


        self.disease_linear=nn.ModuleList([nn.Linear(attention_dim+self.sentence_dim,disease_num[i], dtype=torch.float64) for i in range(category_num)])
        self.disease_norm=nn.ModuleList([nn.LayerNorm(disease_num[i],dtype=torch.float64)for i in range(category_num)])
        self.disease_softmax=nn.ModuleList([nn.Softmax() for i in range(category_num)])
    def attention(self,input,att):
        key = nn.Linear(input.shape[1], input.shape[1],dtype=torch.float64).to(self.device)
        query = nn.Linear(input.shape[1], input.shape[1],dtype=torch.float64).to(self.device)
        value = nn.Linear(input.shape[1], input.shape[1],dtype=torch.float64).to(self.device)
        output=att(key(input),query(input),value(input))
        return output
    def forward(self, gender,pregnancy,profile,description):
        gender=gender.to(self.device)
        pregnancy = pregnancy.to(self.device)
        profile=torch.stack(profile,dim=1).to(self.device)
        profile=F.normalize(profile,dim=0,p=2)
        input=torch.cat([self.gender_emb(gender),self.pregnancy_emb(pregnancy),profile],dim=1)
        input=self.linear1(input)
        attention_output,_ = self.attention(input,self.attention1)

        attention_output = self.layer_norm(attention_output)

        sentence_embeddings=self.sentence_transformer(description)
        all_embedding=torch.cat([attention_output,sentence_embeddings],dim=1)


        category_embedding=self.category_linear(all_embedding)
        category_embedding=self.category_norm(category_embedding)
        category_output=self.softmax(category_embedding)

        # disease_embedding, _ = self.attention(all_embedding, self.attention2)
        disease_embedding=[self.disease_linear[i](all_embedding) for i in range(self.category_num)]
        disease_embedding = [self.disease_norm[i](disease_embedding[i]) for i in range(self.category_num)]
        disease_output=[self.disease_softmax[i](disease_embedding[i]) for i in range(self.category_num)]
        disease_output=torch.stack([F.pad(disease_output[i],(0,self.disease_max_num-disease_output[i].shape[1])) for i in range(len(disease_output))])
        return category_output,disease_output