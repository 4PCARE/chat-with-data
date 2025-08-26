import os
import re
import time
from datetime import datetime

import pandas as pd
import psycopg2
import requests
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import create_sql_query_chain
from langchain.chains.sql_database.query import SQLInputWithTables
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from pymilvus import Collection, connections
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_lottie import st_lottie

from ingestion.function import get_model, get_retrival


def generate_langchain_prompt(
    question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> ChatPromptTemplate:
    
    # Constructing the system message
    if question is not None:
        system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
    else:
        system_msg = "The following is a pandas DataFrame."
        
    if sql is not None:
        system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

    system_msg += f"The following is information about the resulting pandas DataFrame 'data_df': \n{df_metadata}"

    # Create the Langchain prompt with appropriate message types
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'data_df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code.")
    ])
    
    return prompt

def _extract_python_code(markdown_string: str) -> str:
    # Regex pattern to match Python code blocks
    pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

    # Find all matches in the markdown string
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)

    # Extract the Python code from the matches
    python_code = []
    for match in matches:
        python = match[0] if match[0] else match[1]
        python_code.append(python.strip())

    if len(python_code) == 0:
        return markdown_string

    return python_code[0]

def _sanitize_plotly_code(raw_plotly_code: str) -> str:
    # Remove the fig.show() statement from the plotly code
    plotly_code = raw_plotly_code.replace("fig.show()", "")
    # plotly_code = raw_plotly_code
    return plotly_code

# Function to create a single prompt combining the history
def create_conversation_prompt(history):
    """
    Converts the conversation history into a single prompt.
    """
    conversation = ""
    for chat in history:
        conversation += f"Human: {chat['user']}\nAI: {chat['ai']}\n"
    return conversation

# Function to get IP address
def get_ip():
    ip = requests.get("https://api64.ipify.org").text  # Use an external API for IP
    return ip

# Function to fetch user details
def get_user_info():
    ip_data = requests.get("https://api.ipify.org?format=json").json()
    ip_address = ip_data["ip"]
    geo_data = requests.get(f"https://ipapi.co/{ip_address}/json/").json()
    return {"ip": ip_address, "geo": geo_data}

def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip


def get_remote_ip_and_user_agent() -> dict:
    """Get remote IP and User-Agent."""
    
    try:
        # Get the current script run context
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        # Fetch the session information using the session_id
        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None

        # Get the user's IP and User-Agent from the request headers
        user_ip = session_info.request.remote_ip
        user_agent = session_info.request.headers.get("User-Agent", "Unknown")

        return {"ip": user_ip, "user_agent": user_agent}
    
    except Exception as e:
        # Handle any exceptions and return None
        return {"ip": None, "user_agent": None}
    

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def typing_effect(text, delay=0.05):
    """Simulate typing effect for the given text."""
    typed_text = ""
    text_container = st.empty()  # Create an empty container to update dynamically
    for char in text:
        typed_text += char
        text_container.markdown(f"<div style='font-size:18px;'>{typed_text}</div>", unsafe_allow_html=True)
        time.sleep(delay)  # Simulate typing delay
        
# Initialize the environment
def load_environment():
    load_dotenv('.env')
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    if os.getenv("POSTGRES_SCHEMA") is None:
        db_url = f"""postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB_NAME")}"""
    else:
        db_url = f"""postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB_NAME")}?options=-c%20search_path%3D{os.getenv("POSTGRES_SCHEMA")}"""
    
    if os.getenv("POSTGRES_TABLE_LIST") is None:
        db_table = None
    else:
        db_table = os.getenv("POSTGRES_TABLE_LIST").split(",")
        
    return os.getenv("MILVUS_HOST"), os.getenv("MILVUS_PORT"), os.getenv("MILVUS_COLLECTION_NAME"), db_url, os.getenv("POSTGRES_SCHEMA"), db_table

# Setup connections
def setup_connections(milvus_host, milvus_port):
    try: 
        return connections.connect("default", host=milvus_host, port=milvus_port)
    except Exception as e:
        print(f"Error connecting to Milvus: {e} Using standalone mode instead.")
        return connections.connect("default", host='standalone', port=milvus_port)

# Initialize the model
def initialize_model(db_url):
    model = get_model(model_name='kornwtp/simcse-model-phayathaibert', max_seq_length=768)
    con_db = SQLDatabase.from_uri(db_url)
    # llm  = ChatOpenAI(temperature=0, max_tokens=1024, top_p=0.9, model="gpt-3.5-turbo")
    llm = ChatOpenAI(temperature=0, max_tokens=1024, model="gpt-4.1")
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_tokens=1024, top_p=1)
    llms = ChatOpenAI(temperature=0, max_tokens=512, model="gpt-4.1")
    return model, con_db, llm, llms

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []

# Display Sidebar Information
def display_sidebar():
    info = get_remote_ip_and_user_agent()
    if info:
        st.sidebar.markdown(
            f"<div style='font-size:12px;'>"
            f"<strong>IP Address:</strong> {info['ip']}<br>"
            f"<strong>User Agent:</strong> {info['user_agent']}</div>", 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            "<div style='font-size:12px;'>Unable to fetch IP and User-Agent</div>", 
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown(
        """
        <hr style="border-top: 1px solid #ccc;">
        <div style="font-size:12px; color:#888;">
            <strong>Disclaimer:</strong> All interactions on this platform are subject to moderation. 
            Messages and responses may be corrected or adjusted to ensure accuracy and compliance 
            with platform guidelines.
        </div>
        """,
        unsafe_allow_html=True
    )
    
def display_disclaimer():
    st.markdown(
        """
        <hr style="border-top: 1px solid #ccc;">
        <div style="text-align:center; font-size:12px; color:#888;">
            <strong>Disclaimer:</strong> All interactions on this platform are subject to moderation. 
            Messages and responses may be corrected or adjusted to ensure accuracy and compliance 
            with platform guidelines.
        </div>
        """,
        unsafe_allow_html=True
    )

# Set page layout and styles
def set_page_config():
    st.set_page_config(
        page_title="N'4urney Tech Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .main { background-color: #1f1f1f; color: #ffffff; }
        .css-18e3th9 { color: #ffffff; font-family: 'Roboto', sans-serif; }
        .css-184tjsw { background-color: #333333 !important; color: #ffffff !important; }
        .stTitle { color: #00BFFF; font-family: 'Montserrat', sans-serif; }
        .css-1x8cf1d { background-color: #00BFFF !important; color: #ffffff; }
        </style>
        """, unsafe_allow_html=True
    )

# Display chat interface and process user input
def display_chat_interface():
    lottie_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🔮 Chat with N'TIL")

    with col2:
        st_lottie(lottie_animation, height=100)

    user_input = st.chat_input("Ask me a question...")
    return user_input

# Process the chat interaction and generate response
def process_chat(user_input, model, db, db_schema, db_table, collection_name, llm, llms, tracer=None):
    # conversation_history = create_conversation_prompt(st.session_state.history)
    # full_prompt = f"{conversation_history}\nHuman: {user_input}\nAI:"
    callbacks = [tracer] if tracer else None
    
    with st.spinner("4urney is thinking..."):
        query = user_input
        
        
        #Step 1
        trained_sql_df = get_retrival(model, question=query, collection=Collection(collection_name), limit=3)
        # Format SQL examples for prompt
        plain_text_sql = "\n".join(
            f"{i+1}. question: {row['query']}\n   answer: {row['sql']}"
            for i, row in trained_sql_df.iterrows()
        )
        
        data_desc = get_table_and_column_description(db, schema=db_schema)

        #Step 2
        prompt_sql = create_prompt_sql(plain_text_sql, data_desc)
        
        #Step 3
        chain = create_sql_query_chain(llm=llm, db=db, prompt=prompt_sql)
        inputs = {"question": query, "top_k": 10}
        # get response from the sql chain
        if db_table is None: 
            resp_query = chain.invoke(inputs, config={"callbacks": callbacks})
        else: 
            resp_query = chain.invoke(
                                        SQLInputWithTables(inputs, table_names_to_use=['DIM_APPLICATION','DIM_BRANCH_PROVINCE','DIM_BRANCH_REGION','DIM_DATE','DIM_MODEL','DIM_OVERDUE_RANGE','DIM_PROCESS','DIM_PRODUCT','FACT_APPROVAL_PROCESS','FACT_COLLECTION_PERFORMANCE','FACT_LOAN_PERFORMANCE','FACT_SALES'])    
                                        , config={"callbacks": callbacks}
                                    )
        pretty_sql = sqlparse.format(resp_query.replace('```sql', '').replace('```', ''), reindent=True, keyword_case='upper')
        
        #Step 4
        data_df = pd.read_sql(pretty_sql, con=db._engine)

        #Step 5
        prompt_fig = generate_langchain_prompt(question=query, sql=pretty_sql, df_metadata=data_df)
        
        #Step 5.1
        resp_fig = llms.invoke(prompt_fig.format_messages(), config={"callbacks": callbacks}).content
        resp_fig = _sanitize_plotly_code(_extract_python_code(resp_fig))

        # resp_desc = llm.invoke(f"""
        #     โปรดช่วยอธิบายและวิเคราะห์เฉพาะข้อมูลสำคัญจากชุดข้อมูลด้านล่าง:  
        #     {data_df}  
        #     ### บทวิเคราะห์:
        # """).content
        
        #Step 6
        resp_desc = llm.invoke(f"""
            โปรดช่วยวิเคราะห์ชุดข้อมูลด้านล่างโดยสรุปข้อมูลที่สำคัญที่สุด พร้อมทั้งให้ข้อมูลเชิงลึกที่เป็นประโยชน์ที่เกี่ยวเนื่องกับคำถาม:  
            1. ให้คำอธิบายโครงสร้างของข้อมูล (จำนวนแถว, คอลัมน์, และประเภทข้อมูล)
            2. ค้นหาแนวโน้มที่สำคัญหรือรูปแบบที่น่าสนใจ  
            3. ชี้ให้เห็นความสัมพันธ์ระหว่างคอลัมน์ที่เกี่ยวข้อง (ถ้าเป็นไปได้) 
            4. เสนอคำแนะนำหรือข้อสรุปเชิงปฏิบัติการจากข้อมูล  

            คำถาม:
            {query}
            ชุดข้อมูล:  
            {data_df}  

            ### การวิเคราะห์และข้อมูลเชิงลึก:
            """, config={"callbacks": callbacks}).content


        return resp_query, pretty_sql, data_df, resp_fig, resp_desc

def create_prompt_sql(plain_text_sql, data_desc):
    # return PromptTemplate(
    #     input_variables=["input", "table_info", "top_k"],
    #     template=f"""
    #         Given the database schema: {{table_info}}  
    #         and the user input: "{{input}}",  
    #         please generate an SQL query to retrieve the requested information.  

    #         ### Additional Requirements  
    #         - Limit the results to {{top_k}} rows if applicable; otherwise, omit the LIMIT clause

    #         ### Response Guidelines  
    #         1. Generate a **PostgreSQL-compliant** SQL query based on the provided context.  
    #         2. **Avoid explanations**; directly provide the SQL query.  
    #         3. Use only the **most relevant table(s)** and ensure correct referencing of column names.  
    #         4. Ensure the query is **executable** and free from syntax errors.  
    #         5. Adhere strictly to the **PostgreSQL syntax standards**.  

    #         ### Examples  
    #         Here are some examples for reference:  
    #         {plain_text_sql}  

    #         ### Task  
    #         Generate the required SQL query:  

    #         SQL:
    #     """
    # )
    
    # return PromptTemplate(
    # input_variables=["input", "table_info", "top_k"],
    # template=f"""
    #     Given the following **database schema**:  
    #     {{table_info}}  

    #     And the **user query**:  
    #     "{{input}}"  

    #     ### **Requirements**  
    #     - If the user does **not** specify the number of rows, **limit the results to {{top_k}} rows**.  
    #     - If a row limit is specified, **respect the user’s input** and omit unnecessary `LIMIT` clauses.  

    #     ### **Response Format**  
    #     1. **Return only a valid PostgreSQL SQL query**—no explanations.  
    #     2. Use only the **most relevant tables** and ensure correct column references.  
    #     3. The query **must be executable** and free from syntax errors.  
    #     4. Adhere strictly to **PostgreSQL syntax** and best practices.  

    #     ### **Examples**  
    #     Below are examples for reference:  
    #     {plain_text_sql}  

    #     ### **Output**  
    #     Provide the generated SQL query:  

    #     ```sql
    #     -- Your SQL query here
    #     ```
    # """
    # )
    
    return PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template=f"""
                        Given the following **database schema**:  
                        {{table_info}}  
                        
                        **Table & Column descriptions** (if available): 
                        {data_desc}

                        And the **user query**:  
                        "{{input}}"  

                        ### **Requirements**  
                        - If the user does **not** specify the number of rows:
                            - Do not add a `LIMIT` clause if the query is asking for **aggregated results** (e.g., totals, sums, averages, etc.) or if it's asking for **all data**.
                            - Add a `LIMIT` clause **limit the results to {{top_k}} rows** if the query asks for a **top N results** or a **subset of rows** (e.g., "top 10 products", "first 5 records", etc.).
                            - **limit the results to {{top_k}} rows**
                        - If the user specifies a row limit, **use the given limit and exclude any additional `LIMIT` clauses**.  

                        ### **Response Format**  
                        1. **Output only a valid PostgreSQL SQL query—no additional text.**  
                        2. Use only the **most relevant tables** and ensure correct column references.  
                        3. The query **must be executable** and free from syntax errors.  
                        4. Adhere strictly to **PostgreSQL syntax** and best practices.  
                        5. If table is in upper case, please use double quotes to quote the table name. e.g., "FACT_SALES"

                        ### **Examples**  
                        Below are examples for reference:  
                        {plain_text_sql}  

                        ### **Generated SQL Query**  
                        ```sql
                        -- Your SQL query here
                        ```
                    """
                    )

    
def log_historical_conversation(db_url, data_to_upsert):
    # Connect to PostgreSQL
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    # # Define the data you want to upsert
    # data_to_upsert = {
    #     "ip_address": "192.168.1.1",
    #     "user_agent": "Mozilla/5.0",
    #     "user_input": "How do I upsert data?",
    #     "sql_response": "Use the ON CONFLICT clause for PostgreSQL.",
    #     "datetime": datetime.now()
    # }

    # Upsert query (PostgreSQL)
    upsert_query = """
    INSERT INTO "4urney".text2sql_conversation (ip_address, user_agent, user_input, sql_response, datetime)
    VALUES (%(ip_address)s, %(user_agent)s, %(user_input)s, %(sql_response)s, %(datetime)s)
    ON CONFLICT (ip_address, datetime)
    DO UPDATE SET
        user_agent = EXCLUDED.user_agent,
        user_input = EXCLUDED.user_input,
        sql_response = EXCLUDED.sql_response,
        datetime = EXCLUDED.datetime;
    """

    # Execute the upsert query
    cursor.execute(upsert_query, data_to_upsert)

    # Commit the transaction
    connection.commit()

    # Close the connection
    cursor.close()
    connection.close()

    print("Upsert completed successfully.")


def get_table_and_column_description(db, schema=None):
    
    data_desc = ""
    
    if schema is not None:
        query_desc = f"""
            SELECT * FROM poc_til."GET_DATA_DICT"('{schema}', NULL)
            where table_name in
            (
            'DIM_APPLICATION',
            'DIM_BRANCH_PROVINCE',
            'DIM_BRANCH_REGION',
            'DIM_DATE',
            'DIM_MODEL',
            'DIM_OVERDUE_RANGE',
            'DIM_PROCESS',
            'DIM_PRODUCT',
            'FACT_APPROVAL_PROCESS',
            'FACT_COLLECTION_PERFORMANCE',
            'FACT_LOAN_PERFORMANCE',
            'FACT_SALES'
            );
        """
        dict_df = pd.read_sql(query_desc, db._engine)
        
        table_desc = "\n".join(
            f"- Table {row['table_name']}: {row['comment']}"
            for i, row in dict_df[dict_df['desc_level'] == 'table'].iterrows()
        )

        column_desc = "\n".join(
            f"- {row['table_name']}.{row['column_name']}: {row['comment']}"
            for i, row in dict_df[dict_df['desc_level'] == 'column'].iterrows()
        )
        
        data_desc = f"{table_desc}\n{column_desc}"
        
    return data_desc