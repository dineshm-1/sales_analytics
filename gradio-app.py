from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import pyodbc
import pandas as pd
import plotly.express as px
import gradio as gr

# Initialize OpenAI LLM
llm = OpenAI(api_key="your_openai_api_key")

# Define a prompt template to convert natural language to SQL
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a data analyst. Convert the following natural language query into an SQL query.
    The database has a table named 'E2E_Summary' with columns: product_category, sales_amount, and sales_date.
    Query: {query}
    SQL Query:
    """
)

# Initialize LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to connect to SQL Server and execute SQL query
def execute_sql_query(sql_query):
    # SQL Server connection details
    server = 'your_server_name'
    database = 'your_database_name'
    username = 'your_username'
    password = 'your_password'
    
    # Create connection string
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    
    # Connect to SQL Server
    conn = pyodbc.connect(conn_str)
    
    # Execute SQL query and return a DataFrame
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

# Function to generate a chart using Plotly
def generate_chart(df, query):
    if "total sales" in query.lower():
        fig = px.bar(df, x='product_category', y='sales_amount', title='Total Sales by Product Category')
    elif "sales over time" in query.lower():
        df['sales_date'] = pd.to_datetime(df['sales_date'])
        df = df.groupby('sales_date').sum().reset_index()
        fig = px.line(df, x='sales_date', y='sales_amount', title='Sales Over Time')
    else:
        fig = px.bar(df, x='product_category', y='sales_amount', title='Sales Data')
    return fig

# Function to handle user queries
def handle_query(user_query):
    try:
        # Generate SQL query using LangChain and LLM
        sql_query = chain.run(query=user_query)
        
        # Execute SQL query and get data
        df = execute_sql_query(sql_query)
        
        # Generate chart based on the query
        fig = generate_chart(df, user_query)
        
        # Return the chart as a Plotly figure
        return fig
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
def gradio_interface(query):
    fig = handle_query(query)
    if isinstance(fig, str):  # If there's an error, return the error message
        return fig
    return fig

# Create Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your query", placeholder="e.g., Show total sales by product category"),
    outputs=gr.Plot(label="Chart"),
    title="Sales Data Chat Analyzer",
    description="Enter a natural language query to analyze and visualize sales data from the E2E_Summary table."
)

# Launch the Gradio app
iface.launch()