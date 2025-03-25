from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.graphs import StateGraph
from langchain_core.runnables import RunnablePassthrough
from .state_manager import AgentStateManager

class PDFAIAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",  # More capable model
            openai_api_key=openai_api_key
        )
        self.search = DuckDuckGoSearchRun()
        self.tools = self._create_tools()
        self.workflow = self._create_workflow()
        self.state_manager = AgentStateManager()
        self.context = []

    def _create_tools(self) -> List[Tool]:
        return [
            Tool.from_function(
                name="internet_search",
                description="Search the internet for current information",
                func=self.search.run
            ),
            Tool.from_function(
                name="pdf_context",
                description="Get information from the loaded PDF context",
                func=self._get_pdf_context
            )
        ]

    def _get_pdf_context(self, query: str) -> str:
        if not self.context:
            return "No PDF context available."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following PDF context to answer the query:"),
            ("system", "{context}"),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({"context": str(self.context), "query": query})

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(name="pdf_query_workflow")

        # Tool selection prompt
        tool_selector_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the query, decide which tools to use:
            - 'pdf_context': When needing information from the loaded PDF
            - 'internet_search': When needing current or additional information
            - 'both': When needing to combine PDF and internet information
            Output format: {"tool_choice": "pdf_context|internet_search|both"}"""),
            ("user", "{query}")
        ])

        # Define nodes
        tool_selector = (tool_selector_prompt | self.llm | JsonOutputParser())
        
        pdf_context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following PDF context to answer the query:"),
            ("system", "{context}"),
            ("user", "{query}")
        ])
        
        internet_search_prompt = ChatPromptTemplate.from_messages([
            ("system", "Search the internet to supplement the answer:"),
            ("user", "{query}")
        ])
        
        final_synthesis = ChatPromptTemplate.from_messages([
            ("system", "Synthesize information from all sources to provide a complete answer."),
            ("system", "PDF Context: {pdf_result}"),
            ("system", "Internet Search: {search_result}"),
            ("user", "{query}")
        ])

        # Add nodes
        workflow.add_node("tool_selector", tool_selector)
        workflow.add_node("pdf_search", pdf_context_prompt | self.llm)
        workflow.add_node("internet_search", internet_search_prompt | self.llm | self.search)
        workflow.add_node("synthesis", final_synthesis | self.llm)

        def route_tools(x: Dict[str, str]) -> List[str]:
            if x["tool_choice"] == "both":
                return ["pdf_search", "internet_search"]
            elif x["tool_choice"] == "pdf_context":
                return ["pdf_search"]
            else:
                return ["internet_search"]

        workflow.add_conditional_edge(
            "tool_selector",
            route_tools,
            "synthesis"
        )

        return workflow.compile()

    def set_context(self, session_id: str, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Set the PDF context for the agent and persist it."""
        self.context = []
        for doc, meta in zip(documents, metadata):
            self.context.append({
                "content": doc,
                "metadata": meta
            })
        self.state_manager.save_state(session_id, self.context)

    def load_context(self, session_id: str) -> None:
        """Load context from persistent storage."""
        self.context = self.state_manager.load_state(session_id)

    def clear_context(self, session_id: str) -> None:
        """Clear the agent's context."""
        self.context = []
        self.state_manager.clear_state(session_id)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the workflow."""
        try:
            response = await self.workflow.ainvoke({
                "query": query,
                "context": self.context
            })
            
            return {
                "query": query,
                "response": response["synthesis"],
                "sources": {
                    "pdf_context": bool(self.context),
                    "internet_search_used": True  # Now always true in this workflow
                }
            }
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}") 