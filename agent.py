import os
import time
import logging
import string
import re
import random
import ast
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
        trust_remote_code=True,
        pad_token_id=50256
    )
    
    logger.info(f"Modelo carregado em {time.time()-start:.2f}s")
    return generator

template = """[INST]
Você é um expert em Python. Forneça APENAS o código Python funcional e completo.
Não inclua:
- Texto explicativo
- Exemplos de uso
- Blocos markdown

Formato requerido:
1. Código bem estruturado
2. Comentários essenciais
3. Nada além do código

Tarefa: {input}
[/INST]
"""

prompt = PromptTemplate(input_variables=["input"], template=template)

COMMON_QUESTIONS = {
    "quem e voce": "Sou um assistente especializado em gerar código Python funcional e eficiente.",
    "ajuda": "Exemplos de perguntas válidas:\n- Função para calcular média de lista\n- Classe Carro com atributos e métodos",
    "obrigado": "De nada! Fico feliz em ajudar.",
    "o que voce faz": "Gero implementações Python completas para resolver problemas específicos.",
    "sair": "Encerrando a sessão. Até logo! 👋"
}

class CodeAssistant:
    def __init__(self):
        self.generator = load_model()
        self.code_pattern = re.compile(r'(def|class)\s+\w+')
        self.example_pattern = re.compile(r'\[EXEMPL\].*?\[/EXEMPL\]', re.DOTALL)
        self.seen_hashes = set()

    def optimize_response(self, text):
        # Etapa 1: Remoção de tags e exemplos
        text = re.sub(r'.*?\[/INST\]', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = self.example_pattern.sub('', text)
        
        # Etapa 2: Filtragem de linhas únicas
        lines = []
        for line in text.split('\n'):
            clean_line = line.strip()
            if not clean_line:
                continue
                
            # Mantém linhas de código relevantes
            if self.is_code_line(clean_line):
                line_hash = hash(clean_line[:50])  # Considera apenas o início para evitar duplicações
                if line_hash not in self.seen_hashes:
                    lines.append(line)
                    self.seen_hashes.add(line_hash)
        
        cleaned = '\n'.join(lines)
        
        # Validação final
        if not self.is_valid_python(cleaned):
            return self.get_fallback_response(cleaned)
            
        return cleaned.strip()

    def is_code_line(self, line):
        return any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from '])

    def is_valid_python(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            # Permite pequenos erros desde que tenha estrutura básica
            return bool(self.code_pattern.search(code))

    def get_fallback_response(self, partial_code):
        keywords = {
            'função': "Detalhe a funcionalidade. Ex: 'Função para calcular média com tratamento de lista vazia'",
            'classe': "Especifique atributos e métodos. Ex: 'Classe Carro com modelo, ano e método acelerar()'",
            'lista': "Descreva o processamento. Ex: 'Função que recebe lista de números e retorna a soma'"
        }
        
        for key, msg in keywords.items():
            if key in partial_code.lower():
                return f"⚠️ Por favor, {msg}"
        
        return "⚠️ Formule melhor sua pergunta. Ex: 'Função para calcular IMC com parâmetros peso e altura'"

    def normalize_input(self, text):
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
                max_new_tokens=450,
                temperature=0.35,
                top_k=40,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            optimized = self.optimize_response(response[0]['generated_text'])
            logger.info(f"Resposta gerada em {time.time()-start:.2f}s")
            return optimized if optimized else self.get_fallback_response(user_input)
            
        except Exception as e:
            logger.error(f"Erro: {str(e)}")
            return "⚠️ Erro temporário. Tente novamente com uma pergunta mais específica."

assistant = CodeAssistant()
chain = (
    {"input": RunnablePassthrough()}
    | RunnableLambda(assistant.generate)
)

def run_chat():
    print("🚀 Python Code Assistant v3.1")
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