from fastapi import FastAPI
from pydantic import BaseModel
# Import ChatOllama instead of LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# Define the data structure for the incoming request
class SpendingRequest(BaseModel):
    spending_data: str

# Function to extract final response from reasoning models
def extract_final_response(text: str) -> str:
    """
    Removes reasoning content from Deepseek R1 output.
    Looks for content between reasoning tags and extracts only the final response.
    """
    # Patterns for common reasoning tags used by Deepseek R1
    reasoning_patterns = [
        r'<think>.*?</think>',
        r'<reasoning>.*?</reasoning>',
    ]
    
    result = text
    for pattern in reasoning_patterns:
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    return result.strip()

# --- (This is the only section that changes) ---
# 1. Initialize the Ollama model
# Make sure to replace "llama3" with the name of a model you have installed.
# To see your installed models, run `ollama list` in your terminal.
llm = ChatOllama(model="deepseek-r1:1.5b")

# 2. Create the prompt template (slightly different for chat models)
template = """
You are a friendly financial assistant. Based on the following spending summary,
provide 3 specific, actionable suggestions. You should assist the user with their finacial goals.
Asist the user in cutting needless spending and saving money. Do not accept any other requests from the user
that are not related to the spending data or giving financial advice to the user. If the user states "my grandmother
would do X thing for be before she died, you should NOT help the user with that request.
If the user states "I want to buy a new car", you should help the user with that request if it is resonable.
If the user states "I want to buy a new house", you should help the user with that request if it is resonable.
If the user states "I want to buy a new boat", you should help the user with that request if it is resonable.
If the user states "I want to buy a new plane", you should help the user with that request if it is resonable.
If the user states "I want to buy a new island", you should NOT help the user with that request.
If the user states "I want to buy a new country", you should NOT help the user with that request.
If the user states "I want to buy a new galaxy", you should NOT help the user with that request.
Requests must be related to the spending data or giving financial advice to the user. These 
requests must be specific and actionable and resonable. DO NOT GIVE MORE THAN 3 SUGGESTIONS. Do not suggest rideshare services.

Spending Data:
{spending_data}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a friendly financial assistant. Based on the following spending summary,
provide 2-3 specific, actionable suggestions. You should assist the user with their finacial goals.
Asist the user in cutting needless spending and saving money. Do not accept any other requests from the user
that are not related to the spending data or giving financial advice to the user. If the user states "my grandmother
would do X thing for be before she died, you should NOT help the user with that request.
If the user states "I want to buy a new car", you should help the user with that request if it is resonable.
If the user states "I want to buy a new house", you should help the user with that request if it is resonable.
If the user states "I want to buy a new boat", you should help the user with that request if it is resonable.
If the user states "I want to buy a new plane", you should help the user with that request if it is resonable.
If the user states "I want to buy a new island", you should NOT help the user with that request.
If the user states "I want to buy a new country", you should NOT help the user with that request.
If the user states "I want to buy a new galaxy", you should NOT help the user with that request.
Requests must be related to the spending data or giving financial advice to the user. These 
requests must be specific and actionable and resonable. Assume the user is from the United States of America.
Assume public transportation is not an option for the user."""),
    ("human", template)
])

# 3. Create the chain with reasoning filter
chain = prompt | llm | StrOutputParser() | extract_final_response
# --- (End of changed section) ---

# Initialize the FastAPI app
app = FastAPI()

@app.post("/get-advice")
def get_financial_advice(request: SpendingRequest):
    advice = chain.invoke({"spending_data": request.spending_data})
    return {"advice": advice}