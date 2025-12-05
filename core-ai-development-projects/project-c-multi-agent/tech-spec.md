# Technical Specification: Multi-Agent System

## Architecture
```
User Task → Planner Agent → Task Decomposition
                ↓
    [Researcher | Coder | Analyst] Agents
                ↓
            LangGraph State Machine → Tools → Results
```

## Technology Stack
- **Python**: 3.11+
- **Framework**: LangGraph, LangChain
- **LLM**: Ollama (Llama 3, Mistral)
- **Tools**: DuckDuckGo, Python REPL, file system

## Agent Definitions

```python
from langgraph.prebuilt import ToolExecutor
from langchain.agents import AgentExecutor

class ResearcherAgent:
    """Searches web and gathers information"""
    tools = [duckduckgo_search, wikipedia_search]
    
class CoderAgent:
    """Writes and executes code"""
    tools = [python_repl, code_review]
    
class AnalystAgent:
    """Analyzes data and provides insights"""
    tools = [data_analysis, visualization]
    
class PlannerAgent:
    """Decomposes tasks and coordinates agents"""
    tools = [task_decomposition, agent_selector]
```

## LangGraph State Machine

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("analyst", analyst_node)

workflow.add_conditional_edges(
    "planner",
    route_to_agent,
    {
        "researcher": "researcher",
        "coder": "coder",
        "analyst": "analyst",
        "end": END
    }
)

app = workflow.compile()
```

## Tools Implementation

```python
from langchain.tools import Tool

@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=5)
    return "\n".join([r["body"] for r in results])

@tool
def execute_python(code: str) -> str:
    """Execute Python code safely"""
    # Sandboxed execution
    pass

@tool
def read_file(path: str) -> str:
    """Read file contents"""
    with open(path, 'r') as f:
        return f.read()
```

## Project Structure
```
project-c-multi-agent/
├── src/
│   ├── agents/
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   └── analyst.py
│   ├── tools/
│   │   ├── search.py
│   │   ├── code_exec.py
│   │   └── file_ops.py
│   ├── graph.py
│   └── state.py
├── examples/
│   ├── research_task.py
│   ├── coding_task.py
│   └── analysis_task.py
├── app.py
└── README.md
```
