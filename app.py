
import warnings

import streamlit as st

from function import *

warnings.filterwarnings("ignore")

@st.cache_resource
def get_connections(milvus_host, milvus_port):
    return setup_connections(milvus_host, milvus_port)

@st.cache_resource
def get_models(db_url):
    return initialize_model(db_url)

@st.cache_resource
def get_tracer():
    return LangChainTracer(project_name="Chat with Data TIL")

# Main code execution
def main():
    milvus_host, milvus_port, collection_name, db_url, db_schema, db_table = load_environment()
    conn_vec_db = get_connections(milvus_host, milvus_port)
    model, con_db, llm, llms = get_models(db_url)
    tracer = get_tracer()

    set_page_config()
    display_sidebar()
    
    initialize_session_state()
    
    if "initialized" not in st.session_state:
        st.session_state.collection_name = collection_name
        st.session_state.db_schema = db_schema
        st.session_state.db_table = db_table
        st.session_state.model = model
        st.session_state.con_db = con_db
        st.session_state.llm = llm
        st.session_state.llms = llms
        st.session_state.tracer = tracer
        st.session_state.initialized = True
    
    pg = st.navigation([
        st.Page(page_1, title="Chat with Data", icon="💬"), 
        st.Page(page_2, title="Knowledge Training", icon="📚")
    ])
    pg.run()

        
def page_1(): 
    collection_name = st.session_state.collection_name
    db_schema = st.session_state.db_schema
    db_table = st.session_state.db_table
    model = st.session_state.model
    con_db = st.session_state.con_db
    llm = st.session_state.llm
    llms = st.session_state.llms
    tracer = st.session_state.tracer
    # info = get_remote_ip_and_user_agent()
    user_input = display_chat_interface()

    if user_input:
        _, pretty_sql, data_df, resp_fig, resp_desc = process_chat(user_input, model, con_db, db_schema, db_table, collection_name, llm, llms, tracer)
        st.session_state.history.append({
            "user": user_input,
            "ai": resp_desc,
            "fig_code": "import pandas as pd\n\n" + resp_fig,
            "data": data_df,
            'sql': pretty_sql
        })

    for i, chat in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(chat["user"])

        with st.chat_message("assistant"):
            if i == len(st.session_state.history) - 1:
                st.code(chat['sql'])
                if user_input: 
                    typing_effect(chat["ai"], delay=1e-3)
                else: 
                    typing_effect(chat["ai"], delay=0)
            else:
                st.code(chat['sql'])
                st.markdown(chat["ai"], unsafe_allow_html=True)

            try:
                data_df = chat["data"]
                if len(data_df) != 0:
                    _tmp_ = {}
                    exec('import pandas as pd\n\n' + chat["fig_code"], 
                            locals(), 
                            _tmp_)
                    # Check if the expected figure variable (e.g., 'fig') was created
                    if 'fig' in _tmp_:
                        st.plotly_chart(_tmp_['fig'], use_container_width=True, key=f"plot_{i}")  # Display the Plotly chart below the AI response
                        st.success("Graph execution completed successfully.")
                    else:
                        st.warning("No figure was generated from the executed code.")
                else:
                    st.warning("There is no data.")
            except Exception as e:
                st.error(f"Error during graph execution: {e} \n\n Query: \n{pretty_sql} \n\n Fig: \n{resp_fig}")
                
    if user_input:            
        display_disclaimer()
    
    # Auto-scroll to the bottom of the chat container
    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.querySelector('.main');
        if (chatContainer) {
            chatContainer.scrollTo(0, chatContainer.scrollHeight);
        }
        </script>
        """,
        unsafe_allow_html=True
    )
        
def page_2():
    st.header("About")
    st.markdown(
    """
    This application allows users to interact with their database using natural language queries. 
    It leverages advanced language models and vector databases to interpret user input, generate SQL queries, 
    and visualize data insights effectively.
    
    **Key Features:**
    - Natural Language Processing: Understands and processes user queries in plain English.
    - SQL Generation: Automatically generates SQL queries based on user input.
    - Data Visualization: Creates visual representations of data for better insights.
    - Interactive Chat Interface: Engages users in a conversational manner for seamless interaction.
    
    **Technologies Used:**
    - Streamlit: For building the web application interface.
    - LangChain: For managing language model interactions.
    - Milvus: As the vector database for storing and retrieving embeddings.
    - OpenAI GPT-4: For natural language understanding and generation.
    
    **Usage Instructions:**
    1. Enter your database connection details in the sidebar.
    2. Type your query in the chat interface.
    3. View the generated SQL query and data visualization in response.
    
    **Note:** Ensure that your database is accessible and that you have the necessary permissions to execute queries.
    
    For more information, visit our [GitHub repository]
    """
    )
    

if __name__ == "__main__":
    main()