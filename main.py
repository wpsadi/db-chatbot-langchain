import streamlit as st
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI

from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3




os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE', "https://models.github.ai/inference")
model = os.getenv('OPENAI_MODEL', 'openai/gpt-4o')


def connection_string_type(db_choice):
    if db_choice == "PostgreSQL":
        return "postgresql://user:password@localhost/dbname"
    elif db_choice == "MySQL":
        return "mysql://user:password@localhost/dbname"
    elif db_choice == "SQLite":
        return "sqlite:///dbname.db"
    else:
        return None


if "model_name" not in st.session_state:
    st.session_state["model_name"] = "openai/gpt-4.1"
    
    
if "model" not in st.session_state:
    st.session_state["model"] = ChatOpenAI(model=st.session_state["model_name"])
    
if "db_choice" not in st.session_state:
    st.session_state["db_choice"] = None
    
if "connection_string" not in st.session_state:
    st.session_state["connection_string"] = None

if "db_path" not in st.session_state:
    st.session_state["db_path"] = None




st.set_page_config(page_title="LangChain: Chat with DB")


# ask user to choose a database
st.session_state["db_choice"]  = st.sidebar.selectbox(
    "Choose a database",
    ("PostgreSQL", "MySQL", "SQLite"),
)


if st.session_state["db_choice"] is None:
    st.error("Please select a database to continue.")
    st.stop()

# add line break
st.sidebar.markdown("---")
st.sidebar.markdown("### Database Connection")



    
    
# ask for connection string
if st.session_state["db_choice"] == "SQLite":
    st.session_state["db_path"] = st.sidebar.text_input(
        "Enter your SQLite database file path:",
        placeholder=f"e.g. /path/to/your/database.db"
    )
st.session_state["connection_string"] = st.sidebar.text_input(
        "Enter your database connection string:",
        placeholder=f"e.g. {connection_string_type(st.session_state["db_choice"])}"
)



if not st.session_state["connection_string"]:
        st.sidebar.error("Please enter a valid connection string.")
        st.warning("You need to provide a connection string to connect to the database.")
        st.stop()
   
st.sidebar.success(f"You selected: {st.session_state["db_choice"]}")


with st.expander("Database Connection Details", expanded=False):
    st.write(f"**Database Type:** {st.session_state["db_choice"]}")
    st.write(f"**Connection String:** {st.session_state["connection_string"]}")


with st.expander("Model Configuration", expanded=False):

    model = st.selectbox(
    "Choose a model",
    ("openai/gpt-5","openai/gpt-5-chat","openai/gpt-5-mini","openai/gpt-5-nano","openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1", "openai/o1-mini", "openai/o1-preview", "openai/o3", "openai/o3-mini", "openai/o4-mini"),
    on_change=lambda: st.session_state.update({"model": ChatOpenAI(model=model)})
)
    
    
with st.spinner("Setting up the db connection..."):
    if st.session_state["db_choice"] == "SQLite":
        # For SQLite, we use the file path directly
        db_path = st.session_state["db_path"]
        if not db_path:
            st.error("Please provide a valid SQLite database file path.")
            st.stop()
        creator = lambda: sqlite3.connect(db_path)
        db = SQLDatabase(create_engine("sqlite:///", creator=creator))
        
    else:
        # For other databases, we use the connection string
        connection_string = st.session_state["connection_string"]
        if not connection_string:
            st.error("Please provide a valid connection string.")
            st.stop()
        db = SQLDatabase(create_engine(connection_string))
        
toolkit=SQLDatabaseToolkit(db=db,llm=st.session_state["model"])
    
agent=create_sql_agent(
    llm=st.session_state["model"],
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

            

    
# Ask user to input a query
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

        

