import os
import time
import requests
import json
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from Audio_to_text import _Audio_to_text
import streamlit as st
import numpy as np

Qwen_text_url = "PLACE YOUR URL"
Embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class _Audio_Retrieve:
    def __init__(self,Embedding_model = HuggingFaceEmbeddings(model_name=Embedding_model), persist_directory = "./chroma_db",model_url = Qwen_text_url):
        self.Embedding_model = Embedding_model
        self.Chromabd_path = persist_directory
        self.model_url = model_url
        pass

    def load_the_text_file(self, text_file):
        splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_text(text_file)
        return texts

    def Embedding_and_store_(self, file_path):
        embedding_st= time.time()
        audio_to_text= _Audio_to_text()
        text_file = audio_to_text.transcription_(file_path)
        sentence = self.load_the_text_file(text_file)
        document = [Document(page_content=t, metadata = {"source":str(i)}) for i, t in enumerate(sentence)]
        docsearch_store = Chroma.from_documents(documents=document,
                                                persist_directory=self.Chromabd_path,
                                                embedding= self.Embedding_model,
                                                collection_metadata={"hnsw:space":"cosine"}
                                                )
        # persist function is used to store the vector database in local disk
        docsearch_store.persist()
        print(f"the time taken for embedding is {time.time() - embedding_st :.2f}s")
        return docsearch_store # "repetition_penalty":1.2

    def _Qwen_text_call_(self, system_prompt, user_prompt):
        qwen_st = time.time()
        input_data = f"system_prompt:{system_prompt}, user_prompt:{user_prompt}"
        payload = json.dumps({
            "prompt": input_data,
            "temperature": 0.1,
            "max_tokens": 300,
            "top_p": 0.7,
            "top_k": 30,
            "repetition_penalty": 1.2
        })
        headers = {
            'content-Type': 'application/json',
            'ticket': '11059317-57425f24-ac37c5dd80078a80f27cebdfb3dad591'
        }
        response = requests.request(method="POST", url=Qwen_text_url, headers=headers, data=payload)
        response = response.json()
        Generated_text = response['text'][0].replace(input_data, ' ').strip()
        print(f"the taken for generate the text is : {time.time() - qwen_st :.2f}s")
        return Generated_text

    def load_query_for_similarity(self, query, top_similar=3):
        load_st = time.time()
        docsearch = Chroma(
            persist_directory=self.Chromabd_path,
            embedding_function=self.Embedding_model
        )
        retrieved_docs = docsearch.similarity_search_with_score(query, k=top_similar)
        filtered_docs = []
        seen_texts = set()

        for doc, score in retrieved_docs:
            source = doc.metadata.get("source", "Unknown Source")
            if source not in seen_texts:
                filtered_docs.append((doc,score))
                seen_texts.add(source)

        filtered_docs.sort(key=lambda x: x[1],reverse=True)
        st.write("\n**Retrieved Documents with Similarity Scores:**")
        for doc, score in filtered_docs:
            st.write(f"📖**Source**: {doc.metadata.get('source', 'Unknown Source')}, **score**:{score :.4f}")
            st.write(f"📄 **Content Preview**: {doc.page_content[:200]}...\n")

        print(f"Time taken for retrieval: {time.time() - load_st:.2f}s")
        return [doc for doc, _ in filtered_docs]

    def fit_(self, query, top_similar=3):
        docs = self.load_query_for_similarity(query, top_similar=top_similar)

        system_prompt = """
        You are an expert AI assistant that provides precise answers based only on retrieved documents.
        
        ##RULES##:
        1. If the context doesn't contain relevant information, say "No relevant information available"
        2. Do NOT make up or infer information beyond what's in the context
        3. Format any dates, numbers, or proper nouns exactly as they appear in the context
        4. Ensure grammatical correctness and natural sentence flow
        5. Do not include phrases like "based on the context" or "according to the documents"
        6. Start your answer directly without any preamble.
        7. STRICTLY Answer ONLY 2 lines by using information in the provided context
        """
        texts = []
        relevance_scores = []


        query_embedding = self.Embedding_model.embed_query(query)

        for doc in docs:
            doc_text = doc.page_content.strip()
            if not doc_text:
                continue

            doc_embedding = self.Embedding_model.embed_query(doc_text)
            similarity = np.dot(query_embedding, doc_embedding)

            texts.append(doc_text)
            relevance_scores.append(similarity)


        sorted_indices = np.argsort(relevance_scores)[::-1]
        sorted_texts = [texts[i] for i in sorted_indices][:3]

        # st.subheader(sorted_texts)
        context = "\n\n".join(sorted_texts)
        # print ("the context from sorted",context)
        if not context:
            return "No relevant information available."

        user_prompt = f"""Query: {query}?,  context :{context} \n
                Based on the above context only, provide a direct answer to the query"""
        Generated_text = self._Qwen_text_call_(system_prompt=system_prompt, user_prompt=user_prompt)

        return Generated_text.strip()

if __name__=="__main__":
    start_time = time.time()
    Retrieve = _Audio_Retrieve()
    file_path = "PLACE YOUR AUDIO FILE "
    Retrieve.Embedding_and_store_(file_path= file_path)
    query = 'ASK YOUR QUERY'
    retrieve=Retrieve.fit_(query)
    print("ANSWER:",retrieve)
