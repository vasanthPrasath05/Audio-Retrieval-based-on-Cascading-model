import os
import io
import time
import pickle
import numpy as np
import streamlit as st
from pydub import AudioSegment
from Audio_Retireve import _Audio_Retrieve

pickle_file= "data_embedding.pkl"

class App:

    def __int__(self):
        if 'audio' not in st.session_state:
            st.session_state['audio'] = None
        if 'file_name' not in st.session_state:
            st.session_state['file_name']= None
        if 'embeddings' not in st.session_state:
            st.session_state['Embeddings'] = self.load_embedding()

    @st.cache_resource
    def load_the_model(_self,file_name):
        return _Audio_Retrieve(persist_directory=f"./chroma_dp_{file_name}")
    def save_embedding(self):
        embeddings = {
            key: np.array(value).tolist() if isinstance(value, np.ndarray) else value
            for key, value in st.session_state['Embeddings'].items()
        }
        with open(pickle_file,'wb') as f:
            pickle.dump(embeddings,f)

    def load_embedding(self):
        if os.path.exists(pickle_file):
            with open(pickle_file,"rb")as f:
                return pickle.load(f)
    def uploader_(self, files,file_name):

        if files:
            st.write("**Embedding process.**")
            with st.spinner("📥 **Processing...**"):
                time.sleep(2)
                start_time = time.time()
                models_instance = self.load_the_model(file_name)
                Embeddings = models_instance.Embedding_and_store_(file_path=files)
                if "Embeddings" not in st.session_state:
                    st.session_state["Embeddings"] = {}
                st.session_state["Embeddings"][file_name]= np.array(Embeddings).tolist()
                # self.save_embedding()
                st.success(f"✅ Successfully Embedding completed {time.time() - start_time :.2f}s!")
                return  Embeddings
    def Query_(self, query):
            if query:
                with st.spinner("⌛ **wait for response....**"):
                    time.sleep(2)
                    start_time = time.time()
                    models_instance =self.load_the_model(st.session_state['file_name'])
                    Response = models_instance.fit_(query)

                    st.write("💡 **The response**")
                    def _generate_text_stream(Responses):
                        for word in Responses.split():
                            yield  word +" "
                            time.sleep(0.1)
                    st.write_stream(_generate_text_stream(Response))
                    st.success(f"🎉 successfully retrieved time is {time.time() - start_time :.2f}s!")
                    return Response

    def main(self):
        st.markdown("<h1 style='text-align: center;'> AUDIO RAG BASED ON CASCADING MODEL </h1>", unsafe_allow_html=True)
        button = st.radio("Select the Mode",["**Training**", "**Retrieval**🔍"])
        overall_time = time.time()
        if button =="**Training**":
            file = st.file_uploader(" 📁 upload your audio file here (Note: Audio in Mp3 (or) Wav formate only)", type=['wav','mp3'])
            if file:
                file_name = file.name.lower()
                st.session_state['file_name']=file_name
                if file_name.endswith('mp3'):
                    audio = AudioSegment.from_file(file, format="mp3")

                    wav_buffer = io.BytesIO()
                    audio.export(wav_buffer, format="wav")
                    wav_buffer.seek(0)
                    file = wav_buffer
                if st.button("🔊 **Play Audio**"):
                    st.audio(file, format='wav')
                st.session_state['audio'] = file
                embedding = self.uploader_(st.session_state["audio"], file_name)
                if embedding:
                    st.write("## The overall time taken ## ")
                    st.success(f"✅ TIME: {time.time() - overall_time :.2f}s")

            elif file:
                st.session_state['audio'] = file
                file_name= file.name
                if st.button(" 🔊 **play audio**"):
                    st.audio(file,format='audio/wav')
                embedding,_ = self.uploader_(st.session_state["audio"],file_name)
                if embedding:
                    st.write("**The overall time taken** ")
                    st.success(f"✅ TIME: {time.time() - overall_time :.2f}s")

        else:
            if "file_name" not in st.session_state:
                st.session_state["file_name"] = None
            if st.session_state['file_name']:
                st.write("Query and Retrieval")
                query = st.text_input("enter the query to search")
                if st.button("🔍Submit"):
                    query = query.strip()
                    if query is "?":
                        query.replace("?", " ")
                    response = self.Query_(query)

                    if response:
                        st.write("**The overall time taken**")
                        st.success(f"✅TIME: {time.time() - overall_time :.2f}s")
            else:
                st.warning("**NO SAMPLE FILE AVAILABLE**⚠️")


if __name__ == '__main__':
    App_instance  = App()
    App_instance.main()
