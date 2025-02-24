import os
import time
import logging
import string
import re
import random
from transformers import pipeline, set_seed
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from functools import lru_cache
import torch
from accelerate import Accelerator

print(f"PyTorch version: {torch.__version__}")

accelerator = Accelerator()
print(f"Dispositivo: {accelerator.device}")

MODEL_NAME = "Salesforce/codegen-350M-mono"

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
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    logger.info(f"Modelo carregado em {time.time()-start:.2f}s")
    return generator

template = """[INST]
Você é um expert em Python. Forneça APENAS o código Python funcional, completo e comentado.
Não inclua nenhum texto extra, explicações ou formatação.

Tarefa: {input}
[/INST]
"""

prompt = PromptTemplate(input_variables=["input"], template=template)

COMMON_QUESTIONS = {
    "quem e voce": "Sou um assistente especializado em gerar código Python funcional e comentado.",
    "ajuda": "Exemplos de perguntas:\n- Crie uma função para calcular Fibonacci\n- Mostre um exemplo de classe em Python",
    "obrigado": "De nada! Estou aqui para ajudar.",
    "o que voce faz": "Gero códigos Python completos para resolver problemas específicos.",
    "help": "Formule perguntas diretas como:\n- Como fazer X em Python?\n- Exemplo de Y",
    "sair": "Encerrando a sessão. Até logo!"
}

class CodeAssistant:
    def __init__(self):
        self.generator = load_model()
        
    def optimize_response(self, text):
        # Remoção agressiva de elementos do prompt
        text = re.sub(r'\s*\[/?INST\]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\w*\n*', '', text)
        
        # Filtra conteúdo após marcadores-chave
        markers = ["tarefa:", "pergunta:", "resposta:", "código:"]
        for marker in markers:
            if marker in text.lower():
                text = text.split(marker, 1)[-1]
        
        # Remove linhas repetidas e espaços
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Fallback se não gerar código válido
        if 'def ' not in cleaned and 'import ' not in cleaned:
            return "Não consegui gerar um código válido. Tente formular a pergunta de forma mais específica."
        
        return cleaned

    def normalize_input(self, text):
        # Normalização robusta com tratamento de acentos
        text = text.lower().translate(
            str.maketrans(
                'áàâãéèêíìîóòôõúùûç', 
                'aaaaeeeiiioooouuuc',
                string.punctuation + 'ºª'
            )
        ).strip()
        return re.sub(r'\s+', ' ', text)

    def generate(self, user_input):
        try:
            if isinstance(user_input, dict):
                user_input = user_input.get("input", "")
            
            normalized = self.normalize_input(user_input)
            
            if normalized in COMMON_QUESTIONS:
                return COMMON_QUESTIONS[normalized]
                
            full_prompt = prompt.format(input=user_input)
            
            start = time.time()
            response = self.generator(
                full_prompt,
                max_new_tokens=350,
                temperature=0.3,
                top_k=30,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=50256
            )
            
            optimized = self.optimize_response(response[0]['generated_text'])
            logger.info(f"Resposta gerada em {time.time()-start:.2f}s")
            return optimized
            
        except Exception as e:
            logger.error(f"Erro: {str(e)}")
            return random.choice([
                "Posso ajudar com códigos Python. Que tal tentar algo como 'Função para calcular média'?",
                "Vamos tentar novamente? Formule sua pergunta como 'Como fazer X em Python?'",
                "Não entendi completamente. Poderia ser mais específico? Ex: 'Classe para representar um carro'"
            ])

assistant = CodeAssistant()
chain = (
    {"input": RunnablePassthrough()}
    | RunnableLambda(assistant.generate)
)

def run_chat():
    print("🤖 Assistente de Código Python 2.0")
    print("Digite 'ajuda' para orientações ou 'sair' para encerrar\n")
    
    while True:
        try:
            user_input = input("👉 Sua pergunta: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["sair", "exit"]:
                print("\nAté logo! 👋")
                break
                
            print("⚡ Processando...")
            response = chain.invoke(user_input)
            print(f"\n🧠 Resposta:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\nSessão encerrada")
            break

if __name__ == "__main__":
    run_chat()