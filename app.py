from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from example import example
import KEYS
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import functools
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import PromptTemplate

class CustomEmbeddings(HuggingFaceEmbeddings):
    def __call__(self, input):
        return super().__call__(input)

def get_db_chain():

    #create connection with db
    db_user = "root"
    db_password = "sk123456"
    db_host = "localhost"
    db_name = "palm_db"

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3,
    )

    if "GOOGLE_API_KEY" not in os.environ:
         os.environ["GOOGLE_API_KEY"] = KEYS.GOOGLE_API_KEY

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0
    )

    # Define the prompt template
    prompt_template = """
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: SELECT statement without any extra formatting, dont use "```mysql"
    SQLResult: Result of the SQLQuery
    Answer: Final answer after executing the SQLQuery

    Only use the following tables:
    {table_info}

    Question: {input}
    """

    # Create a PromptTemplate object
    prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template=prompt_template
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True,prompt=prompt)
    return chain
    #Initialize embeddings
    # embeddings = CustomEmbeddings(
    #     model_name="sentence-transformers/all-minilm-l6-v2"
    # )
    # to_vectorize = [" ".join(example.values()) for example in example]
    # vectorstore = Chroma.from_texts(to_vectorize,embeddings,metadatas=example)
    #
    # example_selector = SemanticSimilarityExampleSelector(
    #     vectorstore=vectorstore,
    #     k=2,
    # )
    #
    # mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to """
