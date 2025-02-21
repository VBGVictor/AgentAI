import os
import time
import logging
from transformers import pipeline, set_seed
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from functools import lru_cache
import torch
from accelerate import Accelerator

print(f"PyTorch version: {torch.__version__}")

accelerator = Accelerator()
print(f"Dispositivo: {accelerator.device}")

# Mude o modelo para um mais leve
MODEL_NAME = "codellama/CodeLlama-7b-hf"

# Configuração de performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_model():
    logger.info("Carregando modelo...")
    start = time.time()
    
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    logger.info(f"Modelo carregado em {time.time()-start:.2f}s")
    return generator

# 2. Template otimizado para código
template = """[INST] Você é um expert em Python. Forneça apenas código funcional.
Pergunta: {input}
Resposta: [/INST]"""

prompt = PromptTemplate(input_variables=["input"], template=template)

# 3. Sistema de cache para prompts comuns
COMMON_QUESTIONS = {
    "quem é você": "Sou um assistente de programação Python especializado em gerar código funcional.",
    "help": "Formule sua pergunta como: 'Como fazer X em Python?' ou 'Mostre um exemplo de Y'"
}

class CodeAssistant:
    def __init__(self):
        self.generator = load_model()
        
    def optimize_response(self, text):
        # Filtros para respostas melhores
        text = text.split("[/INST]")[-1]  # Pega apenas a resposta
        text = text.split("```")[0]       # Remove markdown extra
        return text.strip()

    def generate(self, user_input):
        user_input = user_input.lower().strip()
        
        # Verifica cache primeiro
        if user_input in COMMON_QUESTIONS:
            return COMMON_QUESTIONS[user_input]
            
        # Otimização de prompt
        full_prompt = f"{prompt.format(input=user_input)}\nCódigo:"
        
        try:
            start = time.time()
            response = self.generator(
                full_prompt,
                max_new_tokens=150,
                temperature=0.3,
                top_k=40,
                num_return_sequences=1,
                do_sample=True
            )
            optimized = self.optimize_response(response[0]['generated_text'])
            logger.info(f"Resposta gerada em {time.time()-start:.2f}s")
            return optimized
            
        except Exception as e:
            logger.error(f"Erro: {str(e)}")
            return "Erro ao processar. Reformule sua pergunta."

# 4. Fluxo otimizado
assistant = CodeAssistant()
chain = (
    {"input": RunnablePassthrough()}
    | RunnableLambda(assistant.generate)
)

# Interface de usuário
def run_chat():
    print("🛠️  Assistente de Código Python")
    print("Digite 'sair' para encerrar\n")
    
    while True:
        try:
            user_input = input("👉 Sua pergunta: ")
            
            if user_input.lower() in ["sair", "exit"]:
                break
                
            print("⚡ Processando...")
            response = chain.invoke(user_input)
            print(f"\n🧠 Resposta:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\nEncerrado pelo usuário")
            break

if __name__ == "__main__":
    run_chat()