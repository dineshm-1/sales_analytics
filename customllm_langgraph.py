from langchain_core.language_models.llms import LLM
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional, Dict, Any, Type, Annotated
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer

# Custom LLM
class CustomLLM(LLM):
    bound_tools: List[BaseTool] = []

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if "employee" in prompt.lower() or "salary" in prompt.lower() or "department" in prompt.lower():
            return json.dumps({
                "tool": "TextToSQL",
                "arguments": {"query": prompt}
            })
        elif "document" in prompt.lower() or "find" in prompt.lower() or "similar" in prompt.lower():
            return json.dumps({
                "tool": "VectorSearch",
                "arguments": {"query": prompt}
            })
        return "I don't know which tool to use."

    @property
    def _llm_type(self) -> str:
        return "custom"

    def bind_tools(self, tools: List[BaseTool], **kwargs: Any) -> Any:
        self.bound_tools = tools

        def wrapped_call(prompt: str, **call_kwargs: Any) -> Dict:
            tool_prompt = prompt + "\nAvailable tools:\n"
            for tool in self.bound_tools:
                tool_prompt += f"- {tool.name}: {tool.description}\n"
                if tool.args_schema:
                    tool_prompt += f"  Args: {tool.args_schema.schema()}\n"

            response = self._call(tool_prompt, stop=call_kwargs.get("stop"), **call_kwargs)

            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
                    return {
                        "tool_calls": [{
                            "name": parsed["tool"],
                            "args": parsed["arguments"]
                        }],
                        "content": None
                    }
            except json.JSONDecodeError:
                tool_call_pattern = r"Use (\w+) with query=(.+)"
                match = re.search(tool_call_pattern, response)
                if match:
                    tool_name, query = match.groups()
                    return {
                        "tool_calls": [{
                            "name": tool_name,
                            "args": {"query": query}
                        }],
                        "content": None
                    }

            return {"content": response, "tool_calls": []}

        return wrapped_call

# TextToSQL Tool
class TextToSQLInput(BaseModel):
    query: str = Field(description="Natural language query to convert to SQL")

class TextToSQLTool(BaseTool):
    name = "TextToSQL"
    description = "Converts natural language queries to SQL and executes them on a PostgreSQL database."
    args_schema: Type[BaseModel] = TextToSQLInput

    def __init__(self):
        super().__init__()
        self.conn = psycopg2.connect(
            dbname="your_db",
            user="your_user",
            password="your_password",
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            sql_query = self._text_to_sql(query)
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            return str(results)
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
        finally:
            self.conn.commit()

    def _text_to_sql(self, query: str) -> str:
        query = query.lower()
        if "employees in" in query:
            department = query.split("in")[-1].strip()
            return f"SELECT * FROM employees WHERE department = '{department}'"
        elif "salary greater than" in query:
            salary = query.split("than")[-1].strip()
            return f"SELECT * FROM employees WHERE salary > {salary}"
        else:
            return "SELECT * FROM employees LIMIT 10"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("TextToSQL does not support async")

    def __del__(self):
        self.cursor.close()
        self.conn.close()

# VectorSearch Tool
class VectorSearchInput(BaseModel):
    query: str = Field(description="Natural language query for vector similarity search")

class VectorSearchTool(BaseTool):
    name = "VectorSearch"
    description = "Performs a vector similarity search on a PostgreSQL vector database using pgvector."
    args_schema: Type[BaseModel] = VectorSearchInput

    def __init__(self):
        super().__init__()
        self.conn = psycopg2.connect(
            dbname="your_db",
            user="your_user",
            password="your_password",
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        register_vector(self.conn)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            query_embedding = self.embedder.encode(query)
            self.cursor.execute(
                "SELECT id, content, embedding <=> %s AS distance FROM documents ORDER BY distance LIMIT 5",
                (np.array(query_embedding),)
            )
            results = self.cursor.fetchall()
            return str([{"id": r["id"], "content": r["content"], "distance": r["distance"]} for r in results])
        except Exception as e:
            return f"Error executing vector search: {str(e)}"
        finally:
            self.conn.commit()

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("VectorSearch does not support async")

    def __del__(self):
        self.cursor.close()
        self.conn.close()

# LangGraph Setup
class AgentState:
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]

llm = CustomLLM()
tools = [TextToSQLTool(), VectorSearchTool()]
tool_map = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

async def call_llm(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    last_message = state.messages[-1]
    response = llm_with_tools(last_message.content)
    return {
        "messages": [AIMessage(
            content=response.get("content", ""),
            tool_calls=response.get("tool_calls", [])
        )]
    }

async def call_tool(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    last_message = state.messages[-1]
    tool_calls = last_message.tool_calls
    results = []
    for tool_call in tool_calls:
        tool = tool_map.get(tool_call["name"])
        if tool:
            result = tool.invoke(tool_call["args"])
            results.append(AIMessage(content=str(result), tool_calls=[]))
        else:
            results.append(AIMessage(content=f"Tool {tool_call['name']} not found"))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# Run the application
async def run_graph():
    queries = [
        "List employees in the sales department",
        "Find documents similar to 'project management best practices'"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        inputs = {"messages": [HumanMessage(content=query)]}
        async for output in graph.astream(inputs):
            for key, value in output.items():
                print(f"Output from {key}:")
                print(value["messages"][-1])
                print("---")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_graph())
