from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools import DuckDuckGoSearchRun
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFAIAgent:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            
      
        self.llm = ChatDeepSeek(
            temperature=0,
            model="deepseek-chat",  
            api_key=api_key
        )
        
        self.search = DuckDuckGoSearchRun()
        self.context = []

    def set_context(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Set the PDF context for the agent."""
        self.context = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(documents, metadata)
        ]

    def clear_context(self) -> None:
        """Clear the agent's context."""
        self.context = []

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using a simple, direct approach."""
        try:
           
            tool_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the query and decide which approach to take.
                Respond with one of these exact words:
                - "pdf" (always use pdf context)
                - "internet" (if internet search is needed)
                - "both" (if both sources are needed)"""),
                ("user", "{query}")
            ])
            
            tool_response = await self.llm.ainvoke(
                tool_prompt.format(query=query)
            )
            
          
            content = tool_response.content.lower()
            if "both" in content:
                tool_choice = "both"
            elif "pdf" in content:
                tool_choice = "pdf"
            elif "internet" in content:
                tool_choice = "internet"
            else:
                tool_choice = "both"  
            
           
            pdf_result = ""
            search_result = ""
            
            # Get PDF context if needed
            if tool_choice in ["pdf", "both"]:
                if self.context:
                    pdf_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Use the following PDF context to answer the query:"),
                        ("system", "{context}"),
                        ("user", "{query}")
                    ])
                    
                    pdf_response = await self.llm.ainvoke(
                        pdf_prompt.format(context=str(self.context), query=query)
                    )
                    pdf_result = pdf_response.content
                else:
                    pdf_result = "No PDF context available."
            
         
            if tool_choice in ["internet", "both"]:
                search_result = self.search.run(query)
            
         
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "Synthesize information from all sources to provide a complete answer."),
                ("system", "PDF Context Information: {pdf_result}"),
                ("system", "Internet Search Information: {search_result}"),
                ("user", "{query}")
            ])
            
            synthesis_response = await self.llm.ainvoke(
                synthesis_prompt.format(
                    pdf_result=pdf_result,
                    search_result=search_result,
                    query=query
                )
            )
            
            return {
                "query": query,
                "response": synthesis_response.content,
                "sources": {
                    "pdf_context": bool(self.context) and tool_choice in ["pdf", "both"],
                    "internet_search_used": tool_choice in ["internet", "both"]
                }
            }
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}") 