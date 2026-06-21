import streamlit as st
import os
import sqlite3
import polars as pl
from typing import Annotated, TypedDict

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# =====================================================================
# UI CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Natural Language DataTable Lookup", layout="wide")

# =====================================================================
# BACKEND INITIALIZATION (Cached for performance)
# =====================================================================
@st.cache_resource(show_spinner="Initializing Database and AI Agents...")
def setup_backend():
    # 1. Use relative paths for GitHub/Render deployment
    csv_path = 'all_energy_statistics.csv' # Ensure this is in the same folder as app.py
    db_path = "mydb.db"

    # Set API Key securely (Locally it uses env, on Render use Environment Variables)
    # Fallback to a placeholder if not set, to prevent instant crashes
    api_key = os.environ.get("GROQ_API_KEY")
    # 2. Load Data via Polars
    if not os.path.exists(db_path):
        df = pl.read_csv(csv_path, infer_schema_length=10000)
        new_columns = {
            col: col.strip().lower().replace(" ", "_").replace("/", "_") 
            for col in df.columns
        }
        df = df.rename(new_columns)
        
        connection_uri = f"sqlite:///{db_path}"
        df.write_database(
            table_name="energy_statistics",
            connection=connection_uri,
            if_table_exists="replace",
            engine="adbc"
        )
    else:
        connection_uri = f"sqlite:///{db_path}"

    db = SQLDatabase.from_uri(connection_uri)

    # 3. Setup LLM & Tools
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # 4. Define Graph
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State):
        messages = state["messages"]
        if len(messages) == 1:
            system_message = {
                "role": "system",
                "content": (
                  "You are an expert SQL assistant. You have access to a SQLite database "
                  "containing energy statistics. Your job is to answer the user's question "
                  "by querying the database. Always check the schema using the appropriate tools "
                  "before executing queries. Provide a final clear answer based on the data found."
                  "Do not write any extra commentary or explanations, just the final answer to the user's question. Your response is directly connected to the SQL tool, therefore, it you do not respond in the correct format, the tool will not be able to execute your query. Always ensure your response is a valid SQL query or a final answer to the user's question."
                  "There are 7 columns in the database: country_or_area, commodity_transaction, year, unit, quantity, quantity_footnotes, category. Use these column names exactly as they are when writing SQL queries."
                )
            }
            messages = [system_message] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# Helper function to extract SQL from LangGraph history
def extract_sql_query(messages):
    # Iterate backwards to find the last SQL query executed by the LLM
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "sql_db_query":
                    return tool_call["args"].get("query")
    return "-- No SQL query was generated for this request."

# Initialize the backend
try:
    app = setup_backend()
except Exception as e:
    st.error(f"Failed to initialize backend. Please check your CSV path and API keys. Error: {e}")
    st.stop()

# =====================================================================
# FRONTEND LAYOUT
# =====================================================================
st.title("⚡ Global Energy Statistics Lookup")

# Informational Context Box
st.info(
    "**About this Dataset:**\n\n"
    "This database contains global energy statistics across multiple countries and years. "
    "It has 7 columns: country_or_area, commodity_transaction, year, unit, quantity, quantity_footnotes, category"
    "You can ask questions about export/import quantities, energy categories (like 'Additives and oxygenates'), "
    "and compare data between different nations." 
    "A sample question could be; 'In which year did Germany generate the highest energy (in units) from Solar photovoltaic cells'"
)

# User Input Form
with st.form("query_form"):
    user_query = st.text_input("Ask a question in plain English:", placeholder="e.g., Which country exported the most additives of oxygen?")
    submitted = st.form_submit_button("Search Database")

# Execution and Results Display
if submitted and user_query:
    with st.spinner("Translating to SQL and searching the database..."):
        
        # Run the graph
        initial_state = {"messages": [{"role": "user", "content": user_query}]}
        final_state = app.invoke(initial_state)
        
        # Extract the outputs
        final_answer = final_state["messages"][-1].content
        generated_sql = extract_sql_query(final_state["messages"])

        st.success("Search Complete!")
        st.divider()

        # Side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 AI Response")
            st.write(final_answer)

        with col2:
            st.subheader("🔍 Generated SQL Query")
            st.code(generated_sql, language="sql")
