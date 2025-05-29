# requirements.txt
"""
langgraph>=0.0.40
langchain>=0.1.0
langchain-openai>=0.1.0
chromadb>=0.4.0
sqlalchemy>=2.0.0
pandas>=2.0.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
python-dotenv>=1.0.0
loguru>=0.7.0
asyncio-throttle>=1.0.0
"""

# .env file template
"""
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///incidents.db
CHROMA_PERSIST_DIRECTORY=./chroma_db
LOG_LEVEL=INFO
"""

# config/settings.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    
    # Database
    database_url: str = "sqlite:///incidents.db"
    
    # Vector Database
    chroma_persist_directory: str = "./chroma_db"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/incident_rag.log"
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Vector Search
    similarity_threshold: float = 0.7
    max_results: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# utils/logger.py
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings

def setup_logger():
    """Configure logging for the application"""
    # Remove default handler
    logger.remove()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        colorize=True
    )
    
    # File handler
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days"
    )
    
    return logger

# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class QueryType(str, Enum):
    INCIDENT_DETAILS = "incident_details"
    RESOLUTION_SEARCH = "resolution_search"
    GENERAL = "general"

class IncidentQuery(BaseModel):
    query: str = Field(..., description="User's query about incidents")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

class IncidentDetails(BaseModel):
    incident_id: str
    title: str
    description: str
    category: str
    priority: str
    status: str
    created_date: datetime
    resolved_date: Optional[datetime] = None
    assigned_to: Optional[str] = None

class ResolutionSuggestion(BaseModel):
    resolution_text: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    source_incident_ids: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)

class QueryResponse(BaseModel):
    query_type: QueryType
    response: str
    incident_details: Optional[List[IncidentDetails]] = None
    resolution_suggestions: Optional[List[ResolutionSuggestion]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

# database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    priority = Column(String(20), nullable=False)
    status = Column(String(50), nullable=False)
    created_date = Column(DateTime, default=func.now(), nullable=False)
    resolved_date = Column(DateTime, nullable=True)
    assigned_to = Column(String(100), nullable=True)
    resolution = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class KnownIssue(Base):
    __tablename__ = "known_issues"
    
    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    resolution_steps = Column(Text, nullable=False)
    confidence_score = Column(Float, default=1.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from config.settings import settings
from database.models import Base
from loguru import logger

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            settings.database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            echo=False
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

db_manager = DatabaseManager()

# services/vector_store.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from loguru import logger
import json

class VectorStoreManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(allow_reset=True)
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections"""
        try:
            self.incidents_collection = self.chroma_client.get_or_create_collection(
                name="incidents_resolutions",
                metadata={"description": "Incident resolutions for similarity search"}
            )
            
            self.known_issues_collection = self.chroma_client.get_or_create_collection(
                name="known_issues",
                metadata={"description": "Known issues and their resolutions"}
            )
            
            logger.info("Vector store collections initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def add_incident_resolution(self, incident_id: str, title: str, 
                                    description: str, resolution: str, 
                                    category: str, metadata: Dict[str, Any] = None):
        """Add incident resolution to vector store"""
        try:
            document = f"Title: {title}\nDescription: {description}\nResolution: {resolution}"
            embedding = await self.embeddings.aembed_query(document)
            
            self.incidents_collection.add(
                documents=[document],
                embeddings=[embedding],
                metadatas=[{
                    "incident_id": incident_id,
                    "title": title,
                    "category": category,
                    "type": "incident_resolution",
                    **(metadata or {})
                }],
                ids=[f"incident_{incident_id}"]
            )
            
            logger.info(f"Added incident resolution to vector store: {incident_id}")
        except Exception as e:
            logger.error(f"Error adding incident resolution: {e}")
            raise
    
    async def add_known_issue(self, issue_id: str, title: str, 
                            description: str, resolution_steps: str, 
                            category: str, metadata: Dict[str, Any] = None):
        """Add known issue to vector store"""
        try:
            document = f"Title: {title}\nDescription: {description}\nResolution Steps: {resolution_steps}"
            embedding = await self.embeddings.aembed_query(document)
            
            self.known_issues_collection.add(
                documents=[document],
                embeddings=[embedding],
                metadatas=[{
                    "issue_id": issue_id,
                    "title": title,
                    "category": category,
                    "type": "known_issue",
                    **(metadata or {})
                }],
                ids=[f"known_issue_{issue_id}"]
            )
            
            logger.info(f"Added known issue to vector store: {issue_id}")
        except Exception as e:
            logger.error(f"Error adding known issue: {e}")
            raise
    
    async def search_similar_resolutions(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar incident resolutions"""
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            
            results = self.incidents_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                if distance <= (1 - settings.similarity_threshold):  # Convert distance to similarity
                    formatted_results.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "rank": i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar resolutions for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar resolutions: {e}")
            return []
    
    async def search_known_issues(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar known issues"""
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            
            results = self.known_issues_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                if distance <= (1 - settings.similarity_threshold):
                    formatted_results.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "rank": i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar known issues for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching known issues: {e}")
            return []

vector_store = VectorStoreManager()

# agents/query_classifier.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from models.schemas import QueryType
from config.settings import settings
from loguru import logger
import re

class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        
        self.classification_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze the following user query and classify it into one of these categories:
            
            1. INCIDENT_DETAILS: Query asking for specific incident information, details, status, or metadata
               Keywords: "show", "find", "get", "details", "status", "when", "who", "what happened"
               
            2. RESOLUTION_SEARCH: Query asking for solutions, fixes, or how to resolve an issue
               Keywords: "how to fix", "solution", "resolve", "repair", "troubleshoot", "similar issue"
               
            3. GENERAL: General questions or unclear intent
            
            User Query: "{query}"
            
            Classification Rules:
            - If asking about specific incident data/information → INCIDENT_DETAILS
            - If asking for help/solutions/resolutions → RESOLUTION_SEARCH  
            - If unclear or general question → GENERAL
            
            Respond with only the classification category name (INCIDENT_DETAILS, RESOLUTION_SEARCH, or GENERAL).
            """
        )
    
    async def classify_query(self, query: str) -> QueryType:
        """Classify user query into appropriate category"""
        try:
            # Simple keyword-based classification as fallback
            query_lower = query.lower()
            
            # Check for incident details keywords
            incident_keywords = [
                "show", "find", "get", "details", "status", "when", "who", 
                "what happened", "incident", "ticket", "id", "number"
            ]
            
            # Check for resolution keywords
            resolution_keywords = [
                "how to fix", "solution", "resolve", "repair", "troubleshoot", 
                "similar", "help", "fix", "solve", "issue"
            ]
            
            incident_score = sum(1 for keyword in incident_keywords if keyword in query_lower)
            resolution_score = sum(1 for keyword in resolution_keywords if keyword in query_lower)
            
            # Use LLM for more sophisticated classification
            try:
                chain = self.classification_prompt | self.llm
                result = await chain.ainvoke({"query": query})
                classification = result.content.strip().upper()
                
                if classification in ["INCIDENT_DETAILS", "RESOLUTION_SEARCH", "GENERAL"]:
                    logger.info(f"Query classified as: {classification}")
                    return QueryType(classification.lower())
            except Exception as e:
                logger.warning(f"LLM classification failed, using keyword-based: {e}")
            
            # Fallback to keyword-based classification
            if incident_score > resolution_score:
                return QueryType.INCIDENT_DETAILS
            elif resolution_score > 0:
                return QueryType.RESOLUTION_SEARCH
            else:
                return QueryType.GENERAL
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return QueryType.GENERAL

# agents/sql_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import Session
from database.connection import db_manager
from database.models import Incident
from models.schemas import IncidentDetails
from config.settings import settings
from loguru import logger
from typing import List, Optional
import sqlalchemy as sa

class TextToSQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        
        self.few_shot_prompt = PromptTemplate(
            input_variables=["query", "schema"],
            template="""
            You are an expert SQL query generator for incident management systems.
            
            Database Schema:
            {schema}
            
            Few-shot Examples:
            
            Q: "Show me incident INC001"
            SQL: SELECT * FROM incidents WHERE incident_id = 'INC001';
            
            Q: "Find all high priority incidents"
            SQL: SELECT * FROM incidents WHERE priority = 'High';
            
            Q: "Show me open incidents assigned to John"
            SQL: SELECT * FROM incidents WHERE status = 'Open' AND assigned_to = 'John';
            
            Q: "Get incidents created in the last 7 days"
            SQL: SELECT * FROM incidents WHERE created_date >= datetime('now', '-7 days');
            
            Q: "Find resolved incidents in network category"
            SQL: SELECT * FROM incidents WHERE category = 'Network' AND status = 'Resolved';
            
            Q: "Show me incidents with resolution containing 'restart'"
            SQL: SELECT * FROM incidents WHERE resolution LIKE '%restart%';
            
            Rules:
            1. Always use proper table and column names from the schema
            2. Use LIKE with % wildcards for text searches
            3. Use proper date functions for date comparisons
            4. Limit results to 50 rows maximum
            5. Only generate SELECT statements
            6. Use single quotes for string literals
            
            User Query: "{query}"
            
            Generate a SQL query to answer this question:
            """
        )
        
        self.schema_info = """
        Table: incidents
        Columns:
        - id (INTEGER, PRIMARY KEY)
        - incident_id (VARCHAR(50), UNIQUE)
        - title (VARCHAR(200))
        - description (TEXT)
        - category (VARCHAR(100))
        - priority (VARCHAR(20)) - Values: Low, Medium, High, Critical
        - status (VARCHAR(50)) - Values: Open, In Progress, Resolved, Closed
        - created_date (DATETIME)
        - resolved_date (DATETIME, nullable)
        - assigned_to (VARCHAR(100), nullable)
        - resolution (TEXT, nullable)
        - created_at (DATETIME)
        - updated_at (DATETIME)
        """
    
    async def generate_sql(self, query: str) -> str:
        """Generate SQL query from natural language"""
        try:
            chain = self.few_shot_prompt | self.llm
            result = await chain.ainvoke({
                "query": query,
                "schema": self.schema_info
            })
            
            sql_query = result.content.strip()
            
            # Basic SQL injection prevention
            if any(keyword in sql_query.upper() for keyword in 
                   ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']):
                raise ValueError("Only SELECT queries are allowed")
            
            # Add LIMIT if not present
            if 'LIMIT' not in sql_query.upper():
                sql_query += ' LIMIT 50'
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    async def execute_query(self, query: str) -> List[IncidentDetails]:
        """Execute natural language query and return incident details"""
        try:
            sql_query = await self.generate_sql(query)
            
            with db_manager.get_session() as session:
                # Execute raw SQL query
                result = session.execute(sa.text(sql_query))
                rows = result.fetchall()
                
                # Convert to IncidentDetails objects
                incidents = []
                for row in rows:
                    incident = IncidentDetails(
                        incident_id=row.incident_id,
                        title=row.title,
                        description=row.description,
                        category=row.category,
                        priority=row.priority,
                        status=row.status,
                        created_date=row.created_date,
                        resolved_date=row.resolved_date,
                        assigned_to=row.assigned_to
                    )
                    incidents.append(incident)
                
                logger.info(f"Found {len(incidents)} incidents")
                return incidents
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

# agents/resolution_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from services.vector_store import vector_store
from models.schemas import ResolutionSuggestion
from config.settings import settings
from loguru import logger
from typing import List

class ResolutionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        
        self.resolution_prompt = PromptTemplate(
            input_variables=["query", "similar_cases", "known_issues"],
            template="""
            You are an expert IT support analyst providing resolution suggestions for incidents.
            
            User Query: "{query}"
            
            Similar Past Incidents:
            {similar_cases}
            
            Known Issues Database:
            {known_issues}
            
            Based on the similar cases and known issues, provide resolution suggestions:
            
            Instructions:
            1. Analyze the user's problem
            2. Review similar past incidents and their resolutions
            3. Check known issues for exact matches
            4. Provide step-by-step resolution suggestions
            5. Rank suggestions by confidence and relevance
            6. Include specific technical steps where applicable
            
            Format your response as structured resolution suggestions with:
            - Clear resolution description
            - Step-by-step instructions
            - Confidence level (0.0 to 1.0)
            - Source incident references
            
            Provide up to 3 best resolution suggestions.
            """
        )
    
    async def suggest_resolutions(self, query: str) -> List[ResolutionSuggestion]:
        """Generate resolution suggestions based on similar cases and known issues"""
        try:
            # Search for similar incident resolutions
            similar_cases = await vector_store.search_similar_resolutions(
                query, n_results=settings.max_results
            )
            
            # Search for known issues
            known_issues = await vector_store.search_known_issues(
                query, n_results=settings.max_results
            )
            
            # Format similar cases for prompt
            similar_cases_text = "\n\n".join([
                f"Case {i+1} (Similarity: {case['similarity_score']:.2f}):\n{case['document']}"
                for i, case in enumerate(similar_cases[:3])
            ]) if similar_cases else "No similar cases found."
            
            # Format known issues for prompt
            known_issues_text = "\n\n".join([
                f"Known Issue {i+1} (Similarity: {issue['similarity_score']:.2f}):\n{issue['document']}"
                for i, issue in enumerate(known_issues[:3])
            ]) if known_issues else "No matching known issues found."
            
            # Generate resolution suggestions
            chain = self.resolution_prompt | self.llm
            result = await chain.ainvoke({
                "query": query,
                "similar_cases": similar_cases_text,
                "known_issues": known_issues_text
            })
            
            # Parse the response into structured suggestions
            suggestions = self._parse_resolution_response(
                result.content, similar_cases, known_issues
            )
            
            logger.info(f"Generated {len(suggestions)} resolution suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating resolution suggestions: {e}")
            return []
    
    def _parse_resolution_response(self, response: str, similar_cases: List, 
                                 known_issues: List) -> List[ResolutionSuggestion]:
        """Parse LLM response into structured resolution suggestions"""
        try:
            suggestions = []
            
            # Simple parsing - in production, you might want more sophisticated parsing
            sections = response.split('\n\n')
            
            for i, section in enumerate(sections[:3]):  # Limit to 3 suggestions
                if section.strip():
                    # Extract source incident IDs
                    source_ids = []
                    if similar_cases:
                        source_ids.extend([
                            case['metadata'].get('incident_id', '')
                            for case in similar_cases[:2]
                        ])
                    
                    # Calculate confidence based on similarity scores
                    confidence = 0.5  # Default
                    if similar_cases:
                        confidence = max(case['similarity_score'] for case in similar_cases[:2])
                    
                    # Extract steps (simple approach)
                    steps = [
                        line.strip() 
                        for line in section.split('\n') 
                        if line.strip() and (line.strip().startswith('-') or line.strip().startswith(f'{i+1}.'))
                    ]
                    
                    suggestion = ResolutionSuggestion(
                        resolution_text=section.strip(),
                        confidence_score=min(confidence, 1.0),
                        source_incident_ids=[id for id in source_ids if id],
                        steps=steps if steps else [section.strip()]
                    )
                    suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error parsing resolution response: {e}")
            return [ResolutionSuggestion(
                resolution_text=response,
                confidence_score=0.5,
                source_incident_ids=[],
                steps=[response]
            )]

# core/workflow.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from agents.query_classifier import QueryClassifier
from agents.sql_agent import TextToSQLAgent
from agents.resolution_agent import ResolutionAgent
from models.schemas import QueryType, QueryResponse, IncidentQuery
from loguru import logger
import time
import asyncio

class WorkflowState(TypedDict):
    query: str
    query_type: QueryType
    incident_details: list
    resolution_suggestions: list
    response: str
    metadata: Dict[str, Any]
    processing_time: float
    confidence: float

class IncidentRAGWorkflow:
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.sql_agent = TextToSQLAgent()
        self.resolution_agent = ResolutionAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify_query", self.classify_query_node)
        workflow.add_node("route_decision", self.route_decision_node)
        workflow.add_node("sql_agent", self.sql_agent_node)
        workflow.add_node("resolution_agent", self.resolution_agent_node)
        workflow.add_node("refine_output", self.refine_output_node)
        
        # Add edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "route_decision")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "route_decision",
            self.route_condition,
            {
                "sql_agent": "sql_agent",
                "resolution_agent": "resolution_agent",
                "refine_output": "refine_output"
            }
        )
        
        workflow.add_edge("sql_agent", "refine_output")
        workflow.add_edge("resolution_agent", "refine_output")
        workflow.add_edge("refine_output", END)
        
        return workflow.compile()
    
    async def classify_query_node(self, state: WorkflowState) -> WorkflowState:
        """Classify the user query"""
        try:
            query_type = await self.query_classifier.classify_query(state["query"])
            state["query_type"] = query_type
            state["metadata"]["classification_time"] = time.time()
            logger.info(f"Query classified as: {query_type}")
            return state
        except Exception as e:
            logger.error(f"Error in classify_query_node: {e}")
            state["query_type"] = QueryType.GENERAL
            return state
    
    async def route_decision_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node for routing"""
        # This is just a pass-through node for routing logic
        return state
    
    def route_condition(self, state: WorkflowState) -> str:
        """Determine which agent to route to"""
        query_type = state["query_type"]
        
        if query_type == QueryType.INCIDENT_DETAILS:
            return "sql_agent"
        elif query_type == QueryType.RESOLUTION_SEARCH:
            return "resolution_agent"
        else:
            return "refine_output"
    
    async def sql_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute SQL agent for incident details"""
        try:
            incidents = await self.sql_agent.execute_query(state["query"])
            state["incident_details"] = [incident.dict() for incident in incidents]
            state["confidence"] = 0.9 if incidents else 0.3
            state["metadata"]["sql_execution_time"] = time.time()
            logger.info(f"SQL agent found {len(incidents)} incidents")
            return state
        except Exception as e:
            logger.error(f"Error in sql_agent_node: {e}")
            state["incident_details"] = []
            state["confidence"] = 0.1
            return state
    
    async def resolution_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute resolution agent for suggestions"""
        try:
            suggestions = await self.resolution_agent.suggest_resolutions(state["query"])
            state["resolution_suggestions"] = [suggestion.dict() for suggestion in suggestions]
            
            # Calculate confidence based on suggestions
            if suggestions:
                avg_confidence = sum(s.confidence_score for s in suggestions) / len(suggestions)
                state["confidence"] = avg_confidence
            else:
                state["confidence"] = 0.2
            
            state["metadata"]["resolution_search_time"] = time.time()
            logger.info(f"Resolution agent found {len(suggestions)} suggestions")
            return state
        except Exception as e:
            logger.error(f"Error in resolution_agent_node: {e}")
            state["resolution_suggestions"] = []
            state["confidence"] = 0.1
            return state
    
    async def refine_output_node(self, state: WorkflowState) -> WorkflowState:
        """Refine and format the final output"""
        try:
            query_type = state["query_type"]
            
            if query_type == QueryType.INCIDENT_DETAILS:
                incidents = state.get("incident_details", [])
                if incidents:
                    state["response"] = self._format_incident_response(incidents)
                else:
                    state["response"] = "No incidents found matching your query."
            
            elif query_type == QueryType.RESOLUTION_SEARCH:
                suggestions = state.get("resolution_suggestions", [])
                if suggestions:
                    state["response"] = self._format_resolution_response(suggestions)
                else:
                    state["response"] = "No resolution suggestions found for your query."
            
            else:
                state["response"] = "I can help you with incident details or resolution suggestions. Please specify what you're looking for."
            
            # Calculate total processing time
            start_time = state["metadata"].get("start_time", time.time())
            state["processing_time"] = time.time() - start_time
            
            logger.info("Output refined successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in refine_output_node: {e}")
            state["response"] = "An error occurred while processing your request."
            state["confidence"] = 0.1
            return state
    
    def _format_incident_response(self, incidents: list) -> str:
        """Format incident details for display"""
        if not incidents:
            return "No incidents found."
        
        response = f"Found {len(incidents)} incident(s):\n\n"
        
        for i, incident in enumerate(incidents[:5], 1):  # Limit to 5 incidents
            response += f"{i}. **{incident['incident_id']}** - {incident['title']}\n"
            response += f"   Status: {incident['status']} | Priority: {incident['priority']}\n"
            response += f"   Category: {incident['category']}\n"
            response += f"   Created: {incident['created_date']}\n"
            if incident.get('assigned_to'):
                response += f"   Assigned to: {incident['assigned_to']}\n"
            response += f"   Description: {incident['description'][:100]}...\n\n"
        
        return response
    
    def _format_resolution_response(self, suggestions: list) -> str:
        """Format resolution suggestions for display"""
        if not suggestions:
            return "No resolution suggestions found."
        
        response = "Here are the recommended resolution approaches:\n\n"
        
        for i, suggestion in enumerate(suggestions, 1):
            confidence_pct = int(suggestion['confidence_score'] * 100)
            response += f"**Suggestion {i}** (Confidence: {confidence_pct}%)\n\n"
            
            if suggestion.get('steps'):
                response += "Steps to resolve:\n"
                for step_num, step in enumerate(suggestion['steps'], 1):
                    clean_step = step.strip().lstrip('-').lstrip(f'{step_num}.').strip()
                    response += f"{step_num}. {clean_step}\n"
            else:
                response += f"{suggestion['resolution_text']}\n"
            
            if suggestion.get('source_incident_ids'):
                response += f"\nBased on similar incidents: {', '.join(suggestion['source_incident_ids'])}\n"
            
            response += "\n" + "="*50 + "\n\n"
        
        return response
    
    async def process_query(self, query_input: IncidentQuery) -> QueryResponse:
        """Process user query through the workflow"""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = WorkflowState(
                query=query_input.query,
                query_type=QueryType.GENERAL,
                incident_details=[],
                resolution_suggestions=[],
                response="",
                metadata={"start_time": start_time},
                processing_time=0.0,
                confidence=0.0
            )
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Create response object
            response = QueryResponse(
                query_type=final_state["query_type"],
                response=final_state["response"],
                incident_details=[IncidentDetails(**incident) for incident in final_state.get("incident_details", [])],
                resolution_suggestions=[ResolutionSuggestion(**suggestion) for suggestion in final_state.get("resolution_suggestions", [])],
                metadata=final_state["metadata"],
                processing_time=final_state["processing_time"],
                confidence=final_state["confidence"]
            )
            
            logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                query_type=QueryType.GENERAL,
                response="An error occurred while processing your request. Please try again.",
                processing_time=time.time() - start_time,
                confidence=0.0
            )

# services/data_loader.py
import pandas as pd
from database.connection import db_manager
from database.models import Incident, KnownIssue
from services.vector_store import vector_store
from loguru import logger
from datetime import datetime
import asyncio
from typing import List, Dict, Any

class DataLoader:
    """Load and initialize data for the RAG system"""
    
    async def load_sample_incidents(self):
        """Load sample incident data"""
        sample_incidents = [
            {
                "incident_id": "INC001",
                "title": "Email server not responding",
                "description": "Users unable to access email server. Server appears to be down.",
                "category": "Email",
                "priority": "High",
                "status": "Resolved",
                "assigned_to": "John Smith",
                "resolution": "Restarted email service and cleared cache. Issue resolved in 30 minutes.",
                "created_date": datetime(2024, 1, 15, 9, 30),
                "resolved_date": datetime(2024, 1, 15, 10, 0)
            },
            {
                "incident_id": "INC002", 
                "title": "Network connectivity issues in Building A",
                "description": "Multiple users in Building A reporting slow internet and connection drops.",
                "category": "Network",
                "priority": "Medium",
                "status": "Resolved",
                "assigned_to": "Sarah Johnson",
                "resolution": "Identified faulty network switch. Replaced switch and restored connectivity.",
                "created_date": datetime(2024, 1, 16, 14, 15),
                "resolved_date": datetime(2024, 1, 16, 16, 45)
            },
            {
                "incident_id": "INC003",
                "title": "Database backup failure",
                "description": "Automated database backup job failed with timeout error.",
                "category": "Database",
                "priority": "Critical",
                "status": "Resolved",
                "assigned_to": "Mike Wilson",
                "resolution": "Increased backup timeout configuration and optimized database indexes. Backup completed successfully.",
                "created_date": datetime(2024, 1, 17, 2, 0),
                "resolved_date": datetime(2024, 1, 17, 8, 30)
            },
            {
                "incident_id": "INC004",
                "title": "Printer offline in Finance department",
                "description": "Main printer in Finance department showing offline status.",
                "category": "Hardware",
                "priority": "Low",
                "status": "Resolved", 
                "assigned_to": "Lisa Brown",
                "resolution": "Reset printer network settings and updated drivers. Printer back online.",
                "created_date": datetime(2024, 1, 18, 11, 20),
                "resolved_date": datetime(2024, 1, 18, 12, 15)
            },
            {
                "incident_id": "INC005",
                "title": "VPN connection failures",
                "description": "Remote users unable to connect to VPN. Authentication errors reported.",
                "category": "Security",
                "priority": "High",
                "status": "Open",
                "assigned_to": "David Chen",
                "resolution": None,
                "created_date": datetime(2024, 1, 20, 8, 45),
                "resolved_date": None
            }
        ]
        
        try:
            with db_manager.get_session() as session:
                for incident_data in sample_incidents:
                    # Check if incident already exists
                    existing = session.query(Incident).filter(
                        Incident.incident_id == incident_data["incident_id"]
                    ).first()
                    
                    if not existing:
                        incident = Incident(**incident_data)
                        session.add(incident)
                        
                        # Add to vector store if resolved
                        if incident_data["resolution"]:
                            await vector_store.add_incident_resolution(
                                incident_id=incident_data["incident_id"],
                                title=incident_data["title"],
                                description=incident_data["description"],
                                resolution=incident_data["resolution"],
                                category=incident_data["category"],
                                metadata={
                                    "priority": incident_data["priority"],
                                    "assigned_to": incident_data["assigned_to"]
                                }
                            )
                
                logger.info(f"Loaded {len(sample_incidents)} sample incidents")
                
        except Exception as e:
            logger.error(f"Error loading sample incidents: {e}")
            raise
    
    async def load_known_issues(self):
        """Load known issues data"""
        known_issues = [
            {
                "issue_id": "KI001",
                "title": "Email Server Performance Issues",
                "description": "Email server becomes unresponsive during peak hours",
                "category": "Email",
                "resolution_steps": "1. Check server memory usage\n2. Restart email service\n3. Clear email cache\n4. Monitor for 1 hour\n5. If persists, increase memory allocation",
                "confidence_score": 0.95
            },
            {
                "issue_id": "KI002",
                "title": "Network Switch Connectivity Problems", 
                "description": "Intermittent network connectivity issues in office buildings",
                "category": "Network",
                "resolution_steps": "1. Check switch LED indicators\n2. Test cable connections\n3. Reboot switch if necessary\n4. Replace switch if hardware failure detected\n5. Update network documentation",
                "confidence_score": 0.90
            },
            {
                "issue_id": "KI003",
                "title": "Database Backup Timeout Errors",
                "description": "Database backup jobs failing due to timeout issues",
                "category": "Database", 
                "resolution_steps": "1. Check database size and growth\n2. Increase backup timeout settings\n3. Optimize database indexes\n4. Schedule backups during off-peak hours\n5. Consider incremental backup strategy",
                "confidence_score": 0.88
            },
            {
                "issue_id": "KI004",
                "title": "Printer Driver and Network Issues",
                "description": "Printers showing offline status or driver problems",
                "category": "Hardware",
                "resolution_steps": "1. Check printer power and network connection\n2. Update printer drivers\n3. Reset printer network settings\n4. Test print from different computer\n5. Replace printer if hardware failure",
                "confidence_score": 0.85
            },
            {
                "issue_id": "KI005",
                "title": "VPN Authentication Failures",
                "description": "Users experiencing VPN connection and authentication issues",
                "category": "Security",
                "resolution_steps": "1. Verify user credentials\n2. Check VPN server status\n3. Test network connectivity\n4. Update VPN client software\n5. Reset user VPN profile if needed\n6. Check firewall rules",
                "confidence_score": 0.92
            }
        ]
        
        try:
            with db_manager.get_session() as session:
                for issue_data in known_issues:
                    # Check if known issue already exists
                    existing = session.query(KnownIssue).filter(
                        KnownIssue.issue_id == issue_data["issue_id"]
                    ).first()
                    
                    if not existing:
                        known_issue = KnownIssue(**issue_data)
                        session.add(known_issue)
                        
                        # Add to vector store
                        await vector_store.add_known_issue(
                            issue_id=issue_data["issue_id"],
                            title=issue_data["title"],
                            description=issue_data["description"],
                            resolution_steps=issue_data["resolution_steps"],
                            category=issue_data["category"],
                            metadata={
                                "confidence_score": issue_data["confidence_score"]
                            }
                        )
                
                logger.info(f"Loaded {len(known_issues)} known issues")
                
        except Exception as e:
            logger.error(f"Error loading known issues: {e}")
            raise
    
    async def initialize_data(self):
        """Initialize all sample data"""
        try:
            await self.load_sample_incidents()
            await self.load_known_issues()
            logger.info("Data initialization completed successfully")
        except Exception as e:
            logger.error(f"Error initializing data: {e}")
            raise

# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models.schemas import IncidentQuery, QueryResponse
from core.workflow import IncidentRAGWorkflow
from services.data_loader import DataLoader
from utils.logger import setup_logger
from loguru import logger
import asyncio

# Setup logging
setup_logger()

# Global workflow instance
workflow_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global workflow_instance
    
    try:
        logger.info("Starting Incident RAG System...")
        
        # Initialize data loader and load sample data
        data_loader = DataLoader()
        await data_loader.initialize_data()
        
        # Initialize workflow
        workflow_instance = IncidentRAGWorkflow()
        logger.info("Incident RAG System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down Incident RAG System...")

# Create FastAPI app
app = FastAPI(
    title="Incident RAG System",
    description="AI-powered incident management system with retrieval-augmented generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_workflow() -> IncidentRAGWorkflow:
    """Dependency to get workflow instance"""
    global workflow_instance
    if workflow_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return workflow_instance

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Incident RAG System is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Incident RAG System",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(
    query: IncidentQuery,
    workflow: IncidentRAGWorkflow = Depends(get_workflow)
):
    """Process user query and return response"""
    try:
        logger.info(f"Processing query: {query.query}")
        response = await workflow.process_query(query)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/incidents/categories")
async def get_incident_categories():
    """Get available incident categories"""
    return {
        "categories": [
            "Email",
            "Network", 
            "Database",
            "Hardware",
            "Security",
            "Software",
            "Infrastructure"
        ]
    }

@app.get("/incidents/priorities")
async def get_incident_priorities():
    """Get available incident priorities"""
    return {
        "priorities": ["Low", "Medium", "High", "Critical"]
    }

@app.get("/incidents/statuses")
async def get_incident_statuses():
    """Get available incident statuses"""
    return {
        "statuses": ["Open", "In Progress", "Resolved", "Closed"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# cli/client.py
import asyncio
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import json

console = Console()

class IncidentRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def query(self, question: str, user_id: str = None, session_id: str = None):
        """Send query to the RAG system"""
        try:
            payload = {
                "query": question,
                "user_id": user_id,
                "session_id": session_id
            }
            
            response = await self.client.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            console.print(f"[red]Error querying system: {e}[/red]")
            return None
    
    def display_response(self, response_data: dict):
        """Display the response in a formatted way"""
        if not response_data:
            return
        
        # Display basic info
        query_type = response_data.get("query_type", "unknown")
        confidence = response_data.get("confidence", 0) * 100
        processing_time = response_data.get("processing_time", 0)
        
        info_text = f"Query Type: {query_type.replace('_', ' ').title()}\n"
        info_text += f"Confidence: {confidence:.1f}%\n"
        info_text += f"Processing Time: {processing_time:.2f}s"
        
        console.print(Panel(info_text, title="Query Information", style="blue"))
        
        # Display main response
        main_response = response_data.get("response", "No response available")
        console.print(Panel(Markdown(main_response), title="Response", style="green"))
        
        # Display incident details if available
        incidents = response_data.get("incident_details", [])
        if incidents:
            table = Table(title="Incident Details")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Priority", style="red")
            table.add_column("Category", style="blue")
            
            for incident in incidents[:5]:  # Limit to 5 for display
                table.add_row(
                    incident.get("incident_id", ""),
                    incident.get("title", "")[:50] + "..." if len(incident.get("title", "")) > 50 else incident.get("title", ""),
                    incident.get("status", ""),
                    incident.get("priority", ""),
                    incident.get("category", "")
                )
            
            console.print(table)
        
        # Display resolution suggestions if available
        suggestions = response_data.get("resolution_suggestions", [])
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                confidence_pct = int(suggestion.get("confidence_score", 0) * 100)
                title = f"Resolution Suggestion {i} (Confidence: {confidence_pct}%)"
                
                suggestion_text = ""
                if suggestion.get("steps"):
                    suggestion_text = "\n".join([f"{j}. {step}" for j, step in enumerate(suggestion["steps"], 1)])
                else:
                    suggestion_text = suggestion.get("resolution_text", "")
                
                console.print(Panel(suggestion_text, title=title, style="yellow"))
    
    async def interactive_mode(self):
        """Run interactive CLI mode"""
        console.print(Panel("Welcome to Incident RAG System CLI", style="bold green"))
        console.print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                question = Prompt.ask("\n[bold blue]Enter your question[/bold blue]")
                
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif not question.strip():
                    continue
                
                console.print("\n[yellow]Processing query...[/yellow]")
                
                response = await self.query(question)
                if response:
                    console.print()
                    self.display_response(response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
    
    def show_help(self):
        """Show help information"""
        help_text = """
        **Available Commands:**
        
        **Incident Details Queries:**
        - "Show me incident INC001"
        - "Find all high priority incidents"
        - "Get open incidents assigned to John"
        - "Show incidents created in the last 7 days"
        
        **Resolution Queries:**
        - "How to fix email server issues?"
        - "Solution for network connectivity problems"
        - "Help with database backup failures"
        - "Troubleshoot printer offline issues"
        
        **General Commands:**
        - `help` - Show this help message
        - `exit` - Quit the application
        """
        
        console.print(Panel(Markdown(help_text), title="Help", style="cyan"))
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

async def main():
    """Main CLI function"""
    client = IncidentRAGClient()
    
    try:
        # Test connection
        response = await client.client.get(f"{client.base_url}/health")
        if response.status_code != 200:
            console.print("[red]Error: Cannot connect to RAG system. Is the server running?[/red]")
            return
        
        await client.interactive_mode()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

# tests/test_workflow.py
import pytest
import asyncio
from core.workflow import IncidentRAGWorkflow
from models.schemas import IncidentQuery, QueryType

@pytest.fixture
def workflow():
    return IncidentRAGWorkflow()

@pytest.mark.asyncio
async def test_incident_details_query(workflow):
    """Test incident details query processing"""
    query = IncidentQuery(query="Show me incident INC001")
    response = await workflow.process_query(query)
    
    assert response.query_type == QueryType.INCIDENT_DETAILS
    assert response.confidence > 0
    assert response.processing_time > 0

@pytest.mark.asyncio
async def test_resolution_query(workflow):
    """Test resolution query processing"""
    query = IncidentQuery(query="How to fix email server issues?")
    response = await workflow.process_query(query)
    
    assert response.query_type == QueryType.RESOLUTION_SEARCH
    assert response.confidence > 0
    assert response.processing_time > 0

@pytest.mark.asyncio 
async def test_general_query(workflow):
    """Test general query processing"""
    query = IncidentQuery(query="Hello, what can you do?")
    response = await workflow.process_query(query)
    
    assert response.query_type == QueryType.GENERAL
    assert "help" in response.response.lower() or "incident" in response.response.lower()

# Docker configuration
"""
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

"""
# docker-compose.yml
version: '3.8'

services:
  incident-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=sqlite:///incidents.db
      - CHROMA_PERSIST_DIRECTORY=/app/chroma_db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./chroma_db:/app/chroma_db
    restart: unless-stopped
"""

# README.md content
"""
# Incident RAG System

A production-grade AI-powered incident management system using LangGraph and Retrieval-Augmented Generation (RAG).

## Features

- **Intelligent Query Classification**: Automatically routes queries to appropriate agents
- **SQL Agent**: Natural language to SQL conversion for incident data retrieval
- **Resolution Agent**: Vector search-based resolution suggestions
- **Production Ready**: Comprehensive logging, error handling, and monitoring
- **FastAPI Backend**: RESTful API with automatic documentation
- **Rich CLI Interface**: Interactive command-line interface

## Architecture

The system uses LangGraph to orchestrate a multi-agent workflow:

1. **Query Classification**: Determines query intent (incident details vs. resolution search)
2. **SQL Agent**: Converts natural language to SQL for incident database queries
3. **Resolution Agent**: Performs vector similarity search for resolution suggestions
4. **Output Refinement**: Formats and enhances responses for presentation

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the API Server**:
   ```bash
   python -m api.main
   ```

4. **Use the CLI Client**:
   ```bash
   python -m cli.client
   ```

## Usage Examples

### Incident Details Queries
- "Show me incident INC001"
- "Find all high priority incidents"
- "Get open incidents assigned to John"

### Resolution Queries  
- "How to fix email server issues?"
- "Solution for network connectivity problems"
- "Help with database backup failures"

## Configuration

Key settings in `config/settings.py`:
- Model configurations (GPT-4, embedding models)
- Vector search parameters
- Database connections
- Logging levels

## Production Deployment

### Docker
```bash
docker-compose up -d
```

### Environment Variables
```bash
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@host:port/db
CHROMA_PERSIST_DIRECTORY=/app/chroma_db
```

## API Documentation

Once running, visit: http://localhost:8000/docs

## Testing

```bash
pytest tests/ -v
```

## License

MIT License
"""
            