"""
ReActMathAgent implementation for mathematical problem solving using LangChain.

This module implements a ReAct (Reasoning + Acting) agent specifically designed
for mathematical problem solving, using LangChain components and chain-of-thought
methodology.
"""

from typing import Any, List, Dict, Optional
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

class MathAgent:
    """
    A ReAct agent specialized for mathematical problem solving.
    
    Implements chain-of-thought reasoning and tool-based problem solving using
    LangChain components.
    """
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        """
        Initialize the ReActMathAgent.
        
        Args:
            llm: Language model to use for reasoning
            tools: List of available mathematical tools
        """
        self.llm = llm
        self.tools = tools
        self.setup_prompts()
        self.setup_chains()
        
    def setup_prompts(self) -> None:
        """Configure prompt templates for different reasoning stages."""
        self.analysis_prompt = PromptTemplate(
            input_variables=["problem"],
            template="""Analyze this mathematical problem:
            {problem}
            
            Break it down into steps and identify key components.
            Thought process:"""
        )
        
        self.tool_selection_prompt = PromptTemplate(
            input_variables=["analysis", "tools"],
            template="""Based on this analysis:
            {analysis}
            
            Available tools:
            {tools}
            
            Which tool would be most appropriate? Why?
            Reasoning:"""
        )
        
        self.reasoning_prompt = PromptTemplate(
            input_variables=["problem", "analysis", "tool_choice"],
            template="""Problem: {problem}
            Analysis: {analysis}
            Selected tool: {tool_choice}
            
            Let's solve this step by step:
            1)"""
        )
        
    def setup_chains(self) -> None:
        """Configure LangChain components and execution chains."""
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt
        )
        
        self.tool_selection_chain = LLMChain(
            llm=self.llm,
            prompt=self.tool_selection_prompt
        )
        
        self.reasoning_chain = LLMChain(
            llm=self.llm,
            prompt=self.reasoning_prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def run(self, problem: str) -> Dict[str, Any]:
        """
        Execute the mathematical problem solving process.
        
        Args:
            problem: Mathematical problem to solve
            
        Returns:
            Dict containing:
                - solution: Final answer
                - steps: List of intermediate reasoning steps
                - tools_used: List of tools utilized
                - confidence: Confidence score in the solution
        """
        try:
            # Track intermediate steps
            steps = []
            
            # Problem analysis
            analysis = self.analysis_chain.run(problem=problem)
            steps.append({"stage": "analysis", "output": analysis})
            
            # Tool selection
            tool_selection = self.tool_selection_chain.run(
                analysis=analysis,
                tools="\n".join([t.name + ": " + t.description for t in self.tools])
            )
            steps.append({"stage": "tool_selection", "output": tool_selection})
            
            # Reasoning and solution
            reasoning = self.reasoning_chain.run(
                problem=problem,
                analysis=analysis,
                tool_choice=tool_selection
            )
            steps.append({"stage": "reasoning", "output": reasoning})
            
            # Execute solution using selected tool
            result = self.agent_executor.run(
                input=reasoning,
                intermediate_steps=steps
            )
            
            return {
                "solution": result,
                "steps": steps,
                "tools_used": self.agent_executor.tools_used,
                "confidence": self.agent_executor.confidence
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "steps": steps,
                "tools_used": [],
                "confidence": 0.0
            }
