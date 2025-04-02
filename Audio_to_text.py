import base64
import json
import requests # type: ignore
import librosa # type: ignore
import soundfile as sf # type: ignore
import numpy as np  #type: ignore
import io
import time

qwen_audio_model = "PLACE YOUR MODEL NAME"
qwen_audio_model_url = "PLACE YOUR URL"
#
system_prompt = """
                You are an advanced speech recognition model.
                Your task is to transcribe spoken content with precision while maintaining proper punctuation and readability.
                
                1. **Start directly with the spoken content**—do not introduce it with phrases like "The speech says" or "Transcribed accurately."
                2. **Do not include timestamps, labels, or formatting markers** unless explicitly requested.
                3. **Preserve sentence structure, punctuation, and speaker intent** while ensuring clarity.
                4. **Do not add any introductory or concluding statements**—provide only the transcribed speech as it was spoken.
                5. **Avoid unnecessary filler words or rephrased summaries**—deliver a faithful representation of the speech.
                """

user_prompt = "Transcribe the following speech exactly as spoken, preserving clarity and punctuation:"

class _Audio_to_text:
    def __init__(self):
        pass
    def encode_audio_to_base64(slef, file_path, duration, overlap):
        """Splits and encodes audio into Base64 without storing it."""
        audio, sample_rate = librosa.load(file_path, sr=None)
        chunk_size = int(duration * sample_rate)
        overlap_size = int((duration - overlap) * sample_rate)

        index = 0
        start_point = 0
        while start_point <len(audio):
            end_point = start_point + chunk_size
            audio_file = audio[start_point: end_point]
            audio_file = np.array(audio_file, dtype= np.float32)
            
            if len(audio) < chunk_size :
                audio_file = np.pad(audio_file,(0, chunk_size - len(audio_file)))
            yield index, audio_file, sample_rate
            
            index +=1
            start_point += overlap_size
            if start_point >= len(audio):
                break
    
    def __base64_audio(self, file_path, duration=30, overlap_second = 5):

        buffer = io.BytesIO()
        audio_base = []

        for _, chunk_audio, sample_rate in self.encode_audio_to_base64(file_path, duration, overlap_second):
            if len(chunk_audio.shape)>2:
                raise ValueError("Audio array must be 1D or 2D.")
            chunk_audio = (chunk_audio * 32767).astype(np.int16)
            buffer.seek(0)
            buffer.truncate(0)
            sf. write(buffer,chunk_audio, sample_rate, format='wav')
            buffer.seek(0)
            audio_bytes = buffer.read()
            audio_b64base = base64.b64encode(audio_bytes).decode("utf-8")
            audio_base.append(audio_b64base)

        return audio_base


    def construct_chat_messages(self,
         system_prompt: str = None, user_prompt: str = None, b64_audio: str = None
        ) -> dict:
        """
        Construct the chat messages to be sent to the chatmodel.

        Args:
            system_prompt (str): The system prompt.
            b64_audio (str): The base64 encoded audio.

        Returns:
            dict: The chat messages to be sent to the chat model.
        """

        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})

        user_content.append(
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/ogg;base64,{b64_audio}",
                },
            }
        )
        messages = []
        if system_prompt:
            system_content = [{"type": "text", "text": system_prompt}]
            messages.append({"role": "system", "content": system_content})

        messages.append({"role": "user", "content": user_content})

        return messages

    def chat_completion(self,messages: dict, sampling_params: dict = None) -> dict:
        if not sampling_params:
            sampling_params = {"temperature": 0, "top_p": 1, "top_k": -1}

        payload = json.dumps(
            {
                "model": qwen_audio_model,
                "messages": messages,

                **sampling_params,
            }
        )
        completion_start_time = time.time()
        response = requests.post(qwen_audio_model_url, data=payload)
        completion_end_time = time.time()
        response_dict = response.json()
        content = response_dict["choices"][0]["message"]["content"]
        token_usage = {
            "prompt": response_dict["usage"]["prompt_tokens"],
            "completion": response_dict["usage"]["completion_tokens"],
        }

        completion_dict = {
            "content": content,
            "token_usage": token_usage,
            "completion_time": round(completion_end_time - completion_start_time, 3),
        }

         return completion_dict['content'].replace("The original content of this audio is:"," ")
    
    def transcription_(self, file_path):
        b64_base = self.__base64_audio(file_path)
        messages = [self.construct_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            b64_audio= audio)
            for audio in b64_base]
        complition = [self.chat_completion(
          messages= msg)
          for msg in messages]
        text_file = '\n'.join(complition)
        return text_file
