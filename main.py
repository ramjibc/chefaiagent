# main script to orchestrate Agent execution
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


RECIPE_DATA = "all_recipes.csv"


def load_recipes():
    """load recipes from input file"""
    recipes = [recipe for recipe in CSVLoader(file_path=RECIPE_DATA).lazy_load()]
    print(f"loaded {len(recipes)} recipes")
    return recipes


def embed_recipes(recipe_docs):
    """Embed the recipe documents and enable vector search"""
    embd = CohereEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(recipe_docs)

    # Add to vectorstore
    return FAISS.from_documents(
        documents=doc_splits,
        embedding=embd,
    )


def create_retriver_tool(vectorized_docs):
    """Take vectorized docs and create a retriever to access them"""
    vectorstore_retriever = vectorized_docs.as_retriever()
    vectorstore_search = create_retriever_tool(
        retriever=vectorstore_retriever,
        name="vectorstore_search",
        description="Retrieve relevant info from a vectorstore that contains documents related to agents, prompt engineering, and adversarial attacks.",
    )
    return vectorstore_search


def create_internet_search_tool():
    """Tavily internet search tool"""
    internet_search = TavilySearchResults()
    internet_search.name = "internet_search"
    internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."

    class TavilySearchInput(BaseModel):
        query: str = Field(description="Query to search the internet with")

    internet_search.args_schema = TavilySearchInput
    return internet_search


def create_chef_agent(vectorstore_search, internet_search):
    """Create the LLM agent"""
    chat = ChatCohere(model="command-r-plus", temperature=0.3)

    # Prompt
    prompt = ChatPromptTemplate.from_template("{input}")

    # Create the ReAct agent
    agent = create_cohere_react_agent(
        llm=chat,
        tools=[internet_search, vectorstore_search],
        prompt=prompt,
    )
    return agent


def execute():
    """Create and execute the chef agent"""

    # Preamble
    preamble = """
    You are a master chef and personal cook who can suggest dishes given some food ingredients and dietary restrictions.
    You are equipped with an internet search tool and a special vectorstore of information about dishes as well as ingredients and recipes to cook those dishes.
    If the query covers the topics of dishes and recipes, use the vectorstore search. 
    For food allergy and dietary information, use the internet search.
    If you find a suitable recipe, return ONLY the ingredients as output. INCLUDE ingredient name and quantity.
    Output should be table with the following format. 
    | Ingredient | Quantity | 
    | ---------- | -------- |
    """
    vectorized_recipes = embed_recipes(load_recipes())
    retriever_tool = create_retriver_tool(vectorized_docs=vectorized_recipes)
    internet_tool = create_internet_search_tool()
    agent = create_chef_agent(
        vectorstore_search=retriever_tool, internet_search=internet_tool
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=[internet_tool, retriever_tool], verbose=True
    )

    agent_executor.invoke(
        {
            "input": "I would like to eat something with pecan in it and savory",
            # "input": "Is falafel gluten free?",
            "preamble": preamble,
        }
    )


if __name__ == "__main__":
    execute()
