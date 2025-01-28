import gradio as gr
import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text
from langchain import PromptTemplate
from langchain.llms import OpenAI
import matplotlib.pyplot as plt
from io import BytesIO

# Database connection setup
server = "your_server"
database = "your_database"
username = "your_username"
password = "your_password"  # Use Windows authentication or provide credentials
connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)

# LLM Setup (e.g., OpenAI GPT)
llm = OpenAI(temperature=0)

# Function to query the database
def query_database(sql_query):
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(sql_query), conn)
        return result
    except Exception as e:
        return f"Error: {e}"

# Function to analyze user input and generate SQL query
def analyze_chat_and_generate_chart(user_input):
    # Template to guide the LLM
    template = """
    You are an expert data analyst. Generate an SQL query based on the following user request:
    "{user_request}"
    The SQL query should fetch data from the table 'E2E_Summary'. Focus only on valid columns and ensure it's syntactically correct.
    """
    prompt = PromptTemplate(input_variables=["user_request"], template=template)
    sql_query = llm(prompt.format(user_request=user_input)).strip()
    
    try:
        # Execute the query
        df = query_database(sql_query)
        if isinstance(df, str):  # Check for errors
            return df

        # Create a chart
        fig, ax = plt.subplots(figsize=(8, 5))
        df.plot(ax=ax, kind='bar', legend=True)
        plt.title("Data Analysis Result")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.tight_layout()

        # Save the chart to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf, sql_query
    except Exception as e:
        return f"Error while generating chart: {e}", sql_query

# Gradio Interface
def chat_analytics(user_input):
    chart, sql_query = analyze_chat_and_generate_chart(user_input)
    if isinstance(chart, str):  # Return error message
        return chart, None
    return chart, f"Generated SQL Query:\n{sql_query}"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Chat Analytical Application")
    gr.Markdown("Interact with the system by entering your data analysis requests.")
    
    with gr.Row():
        user_input = gr.Textbox(label="Your Query (e.g., Show sales by region)")
        submit_btn = gr.Button("Analyze")
    
    with gr.Row():
        chart_output = gr.Image(label="Generated Chart")
        sql_output = gr.Textbox(label="Generated SQL Query")
    
    submit_btn.click(chat_analytics, inputs=[user_input], outputs=[chart_output, sql_output])

# Run the app
app.launch()
