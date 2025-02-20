import os
from transformers import pipeline, set_seed
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompt_values import StringPromptValue  # Adicionar esta importação
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Desabilitar avisos
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

set_seed(42)

logger.info("Carregando o modelo...")
code_generator = pipeline(
    "text-generation", 
    model="gpt2",
    device=-1
)

# Template modificado para melhor performance
template = """Você é um assistente de programação Python. 
Siga estas regras:
1. Responda APENAS com código válido ou explicações técnicas
2. Formate o código com markdown
3. Seja conciso

Tarefa: {input}"""

prompt = PromptTemplate(input_variables=["input"], template=template)

class CodeLLM:
    def __init__(self, generator):
        self.generator = generator
        
    def generate(self, prompt_value):
        # Converter StringPromptValue para texto
        if isinstance(prompt_value, StringPromptValue):
            text = prompt_value.text
        else:
            text = str(prompt_value)
        
        logger.debug(f"Prompt recebido: {text}")
        
        response = self.generator(
            text,
            max_new_tokens=200,
            temperature=0.4,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=50256
        )
        return response[0]['generated_text'].replace(text, "").strip()

# Chain corrigida
code_llm = CodeLLM(code_generator)
chain = (
    {"input": RunnablePassthrough()}
    | prompt
    | RunnableLambda(code_llm.generate)
)

# Loop de interação atualizado
def run_chat():
    logger.info("Agent pronto! Digite 'sair' para encerrar.")
    
    while True:
        try:
            user_input = input("\n👨💻 Usuário: ")
            
            if user_input.lower() in ["sair", "exit"]:
                logger.info("Encerrando...")
                break
                
            if not user_input.strip():
                print("⚠️ Digite um comando válido")
                continue
                
            print("\n🤖 Processando...")
            response = chain.invoke(user_input)
            print(f"\n🧠 Resposta:\n{response}")
            
        except KeyboardInterrupt:
            logger.info("Encerrado pelo usuário")
            break
        except Exception as e:
            logger.error(f"Erro detalhado: {str(e)}", exc_info=True)
            print("❌ Erro ao processar. Tente novamente.")

if __name__ == "__main__":
    run_chat()