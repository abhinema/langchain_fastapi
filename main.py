from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langserve.schema import CustomUserType
#from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Langchain LLM API",
    version="1.0",
    description="A Multi-LLM API with Langchain"
)

# Define Prompts
prompt = ChatPromptTemplate.from_template("Give me detailed information on {topic} in 150 words.")


# Initialize LLM Models
llm = Ollama(model="llama3.2")
#llm = Ollama(model= {model} )

# Create API Routes with Langchain Pipelines
add_routes(app, prompt | llm, path="/inference")




class Foo(CustomUserType):
    bar: int


def model_list(version: str) -> str:
    """Sample function that expects a Foo type which is a pydantic model"""
    
    return "Ollama"


# Note that the input and output type are automatically inferred!
# You do not need to specify them.
# runnable = RunnableLambda(func).with_types( # <-- Not needed in this case
#     input_type=Foo,
#     output_type=int,
#
add_routes(app, RunnableLambda(model_list), path="/model_list")




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8081)