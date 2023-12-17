from transformers import pipeline
from ctransformers import AutoModelForCausalLM
#from basic_chatbot import *


class DigitalAssistant:
    def __init__(self, translator_model_ru_to_en, translator_model_en_to_ru, device=0):
        self.translator_ru_to_en = pipeline("translation", model=translator_model_ru_to_en, device=device)
        self.translator_en_to_ru = pipeline("translation", model=translator_model_en_to_ru, device=device)
        self.assistant_model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)
        self.device = device

    def process_text(self, text):
        # Translate text to English
        translated_text_en = self.translator_ru_to_en(text, max_length=800)[0]['translation_text']
        print("translated_text_en:",translated_text_en)
        #basic_response = get_response(predict_class(translated_text_en), intents)
        #print("basic_response:", basic_response)

        # Prepare text for the assistant model
        #prompt = f"<s>[INST] You are an assistant to the Mirniy city administration [/INST] Ok, I will answer only relevant questions. [INST] {translated_text_en}, The basic response is {basic_response}. Answer: [/INST]"
        prompt = f"<s>[INST] You are an assistant to the Mirniy city administration [/INST] Ok, I will answer only relevant questions. [INST] {translated_text_en}  Answer: [/INST]"
        # Generate response
        response_text_en = self.assistant_model(prompt, max_new_tokens=256+128, temperature=0.7)

        # Translate response back to Russian
        response_text_ru = self.translator_en_to_ru(response_text_en, max_length=1000)[0]['translation_text']
        response_text_ru = self.replace_text(response_text_ru)

        return response_text_ru
    
    @staticmethod
    def replace_text(response_text_ru):  
        replacements = {  
            "администрации Мирни": "администрации города Мирный",  
            "администрация Мирни": "администрация города Мирный",  
            "администрацией Мирни": "администрацией города Мирный",  
            "города Мирни": "города Мирный",  
            "Украина": " ",  
            "в Мирни": "в Мирный",  
            "городе Мирни": "городе Мирный",  
            "городом Мирни": "городом Мирный",  
            "в Мирныйе": "в городе Мирный",  
            "мэра Мирни": "мэра города Мирный",  
            "мэре Мирни": "мэре города Мирный",  
            "Мирныййском городе": "городе Мирный"  
        }  

        for old, new in replacements.items():  
            response_text_ru = response_text_ru.replace(old, new)  

        return response_text_ru  


assistant = DigitalAssistant("Helsinki-NLP/opus-mt-ru-en", "Helsinki-NLP/opus-mt-en-ru")