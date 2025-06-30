import os
import re
import fitz
import nltk
import torch
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer,BertModel

nltk.download('punkt')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

load_dotenv()
api_key=os.getenv("API_KEY")


class DocLoader:
    def __init__(self,path):
        self.path=path
        self.pages=[]
        self.chunks=[]
        self.page_num=[]
        self.load()

    def load(self):
        doc=fitz.open(self.path)
        for page_num in range(len(doc)):
            page=doc.load_page(page_num)
            text=page.get_text()
            self.pages.append(text)
            #p.append(page_num)
        doc.close()

    def preprocess(self):
        placeholder = "___PARAGRAPH_BREAK___"
        for i in range(len(self.pages)):
            self.pages[i]=re.sub(r'\n\d\n+',r'\n',self.pages[i])  #removes page numbers
            self.pages[i]=re.sub(r'(\n )+',placeholder,self.pages[i])
            self.pages[i]=re.sub(r'\n',' ',self.pages[i])
            self.pages[i]=re.sub(placeholder,'\n',self.pages[i])
        
    def split(self,max_length=150):
        for i,page in enumerate(self.pages):
            sentences = sent_tokenize(page)
            current=''
            for sent in sentences:
                if len(current.split())+len(sent.split()) <max_length:
                    current+=' '+sent
                else:
                    self.chunks.append(current)
                    self.page_num.append(i+1)
                    current=sent
            if current:
                self.chunks.append(current.strip())
                self.page_num.append(i+1)
        return self.chunks, self.page_num
    
class Embedder:
    def __init__(self,model_name="bert-base-uncased"):
        self.DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer=BertTokenizer.from_pretrained(model_name)
        self.bert_model=BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.to(self.DEVICE)
        self.tokens=[]

    def tokenize(self,chunks):
        self.tokens=self.tokenizer.batch_encode_plus(chunks,add_special_tokens=True,padding=True,truncation=True)
        input_ids=torch.tensor(self.tokens["input_ids"]).to(self.DEVICE)
        mask=torch.tensor(self.tokens["attention_mask"]).to(self.DEVICE)
        return input_ids,mask
    
    def mean_embedding(self,input_ids,attention_masks):
        mean_emb=[]
        for id,mask in zip(input_ids,attention_masks):
            id=id.unsqueeze(0)
            mask=mask.unsqueeze(0)
            with torch.no_grad():
                emb=self.bert_model(id,mask)[0].squeeze(0)

                valid_mask=mask[0]==1
                valid_emb=emb[valid_mask,:]
                mean_embedding = valid_emb.mean(dim=0)

                mean_emb.append(mean_embedding.unsqueeze(0))
        aggregated_mean_emb=torch.cat(mean_emb)
        return aggregated_mean_emb
        
class Retriever:
    def __init__(self,embedder):
        self.embedder=embedder
        self.chunks=[]
        self.page_num=[]
        self.embeddings=[]
        self.index=None
        self.dim=768

    def build_index(self,chunks,page_num):
        self.chunks=chunks
        self.page_num=page_num
        input_ids,mask=self.embedder.tokenize(chunks)
        self.embeddings=self.embedder.mean_embedding(input_ids,mask)
        np_aggr_mean_emb=self.embeddings.cpu().numpy().astype('float32')
        self.index=faiss.IndexFlatL2(self.dim)
        self.index.add(np_aggr_mean_emb)

    def retrieve(self,query,k=7):
        tokenizer=self.embedder.tokenizer
        bert_model=self.embedder.bert_model
        query_tokens=tokenizer.encode(query,add_special_tokens=True,return_tensors='pt')
        query_emb=bert_model(query_tokens)[0].squeeze(0)
        query_emb=query_emb.mean(dim=0).unsqueeze(0)
        query_emb_np=query_emb.cpu().detach().numpy().astype('float32')
        D,I=self.index.search(query_emb_np,k)
        retrieved_chunks=[]
        retrieved_pages=[]
        for i in I[0]:
            retrieved_chunks.append(self.chunks[i])
            retrieved_pages.append(self.page_num[i])
        return retrieved_chunks,retrieved_pages
    
class AnswerGenerator:
    def __init__(self,model_name='gemini-1.5-flash-latest',temp=0.7,max_output_tokens=1024,api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(temperature=temp,max_output_tokens=max_output_tokens)
        self.prompt=[]
        self.response=[]

    def generate(self,query,chunks,page_num):
        context="Context: "
        for i,(chunk,page) in enumerate(zip(chunks,page_num)):
            context+=f"\nChunk : {chunk}\n page number of chunk  is : {page}"

        self.prompt=f"""You are a helpfull assistant. Your task is to answer the question based on the given context and their corresponding page numbers.
        - Cite relevant page numbers to support your answer, but group citations logically rather than repeating them after every sentence.
        - When multiple sentences refer to the same context, place the page number once at the end of the related section.
        - Provide detailed and accurate explanations wherever applicable.
        - Do not make up answer if the answer is not found in context.
        {context}
        Question:
        {query}
        Answer:
        """
        
        try:
            
            self.response=self.gen_model.generate_content(self.prompt, generation_config=self.generation_config)
            return self.response.text
        except Exception as e:
            print("Error occured : ",e)
            return None
        

if __name__=="__main__":

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    load_dotenv()
    api_key=os.getenv("API_KEY")

    embedder=Embedder()
    ret=Retriever(embedder)

    doc=DocLoader(r"C:\Users\91859\Desktop\rag_scratch\rag\sample_pdf.pdf")
    doc.preprocess()
    chunks,page_num=doc.split()

    ret.build_index(chunks,page_num)
    query="Explain the encoder and decoder stacks of transformers"
    retrieved_chunks,retrieved_pages=ret.retrieve(query)

    generator=AnswerGenerator(api_key=api_key)
    generator.generate(query,retrieved_chunks,retrieved_pages)
    print(generator.response.text)