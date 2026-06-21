# DataTalk: Natural Language Lookup on Global Energy Statistics

<div>
  <a href="https://www.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-A473E8?style=for-the-badge" alt="LangChain" />
  </a>
  <a href="https://pola.rs/">
    <img src="https://img.shields.io/badge/Polars-1D2B3A?style=for-the-badge" alt="Polars" />
  </a>
  <a href="https://arrow.apache.org/">
    <img src="https://img.shields.io/badge/PyArrow-E04E14?style=for-the-badge&logo=apache&logoColor=white" alt="PyArrow" />
  </a>
  <a href="https://www.sqlite.org/">
    <img src="https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  </a>
  </a>
  <a href="https://groq.com/">
    <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge" alt="Groq" />
  </a>
  <a href="https://langchain-ai.github.io/langgraph/">
    <img src="https://img.shields.io/badge/LangGraph-A473E8?style=for-the-badge" alt="LangGraph" />
</div>

## Tech Stack & Components

* **Frontend (Streamlit):** Handles the UI, state caching, and responsive data presentation.
* **Data Processing (Polars & PyArrow):** Used to compress the initial 135 MB CSV into a highly optimized Parquet file (2.5 MB), bypassing GitHub file limits while strictly enforcing column data types for stability.
* **Database (SQLite):** Acts as the localized relational database, rapidly ingesting the Parquet data via a high-speed ADBC driver.
* **Agent Orchestration (LangGraph):** Manages the looping, stateful execution flow between the LLM and the database tools.
* **LLM Engine (Groq & LangChain):** Connects the application to the `llama-3.3-70B-versatile` model, enabling near-instantaneous reasoning and precise Text-to-SQL translation.
* **Deployment (Render):** Cloud-hosted.

## The Agentic Workflow

This project leverages an autonomous agent architecture rather than rigid, pre-defined data pipelines. Here is how the AI seamlessly navigates the database:

1. **State Initialization:** The user submits a natural language query via the Streamlit frontend. The prompt is initialized into a LangGraph `StateGraph`.
2. **Schema Inspection:** The LLM (Llama 3.3) evaluates the request and determines it needs context. It issues a tool call to inspect the database schema. LangGraph routes the execution to the Tool Node, which queries SQLite and returns the column structures.
3. **Query Generation:** Armed with the schema, the LLM constructs a syntactically correct SQL query mapping the user's intent to specific columns, filters, and superlatives.
4. **Execution & Validation:** The agent executes the generated SQL via the `sql_db_query` tool. If the query fails, the agent can read the error and correct its own SQL syntax in a loop.
5. **Final Synthesis:** Once the raw data is returned from the database, the LLM synthesizes it into a final, human-readable response and terminates the graph execution.

