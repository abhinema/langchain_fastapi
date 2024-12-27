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
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Langchain LLM API",
    version="1.0",
    description="A Multi-LLM API with Langchain"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Define Prompts
prompt = ChatPromptTemplate.from_template("Give me detailed information on {topic} in 150 words.")


# Initialize LLM Models
llm = Ollama(model="llama3.2")
#llm = Ollama(model= {model} )



async def chat(self, request: Request):
"""
Generate a chat response using the requested model.
"""

# Passing request body JSON to parameters of function _chat
# Request body follows ollama API's chat request format for now.
params = await request.json()
self.logger.debug("Request data: %s", params)

chat_response = self._client.chat(**params)

# Always return as streaming
if isinstance(chat_response, Iterator):
    def generate_response():
        for response in chat_response:
            yield json.dumps(response) + "\n"
    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
elif chat_response is not None:
    return json.dumps(chat_response)


# Create API Routes with Langchain Pipelines
add_routes(app, prompt | llm, path="/inference")



############################################

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
#########################################



if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.120", port=8081)
