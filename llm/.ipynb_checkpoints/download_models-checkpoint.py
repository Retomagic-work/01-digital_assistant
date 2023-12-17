#import torch
from transformers import pipeline
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
class DummyDigitalAssistant:
    def __init__(self, translator_model_ru_to_en, translator_model_en_to_ru, device=0):
        self.translator_ru_to_en = pipeline("translation", model=translator_model_ru_to_en, device=device)
        self.translator_en_to_ru = pipeline("translation", model=translator_model_en_to_ru, device=device)
        self.assistant_model =AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)
        self.device = device

assistant = DummyDigitalAssistant("Helsinki-NLP/opus-mt-ru-en", "Helsinki-NLP/opus-mt-en-ru", device="cpu")
