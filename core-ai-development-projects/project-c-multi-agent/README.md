# Project C: Multi-Agent AI System

## Objective

Build a sophisticated multi-agent system using LangGraph where specialized agents (Planner, Researcher, Coder, Analyst) coordinate to solve complex tasks through a state machine workflow.

**What You'll Build**: A production-ready multi-agent system with a Planner agent that decomposes tasks and routes them to specialized agents (Researcher for web search, Coder for code execution, Analyst for data analysis), all coordinated through a LangGraph state machine with tool calling and memory.

**What You'll Learn**: Multi-agent architectures, LangGraph state machines, agent coordination patterns, tool calling, task decomposition, agent communication, and building complex agentic AI systems.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hours 1-2**: Setup LangGraph and dependencies (install, Ollama models, test)
- **Hours 3-5**: Implement basic agents (Planner, Researcher, Coder with prompts)
- **Hours 6-7**: Create tool ecosystem (web search, code execution, file operations)
- **Hour 8**: Build LangGraph state machine (nodes, edges, routing)

### Day 2 (8 hours)
- **Hours 1-2**: Add Analyst agent (data analysis tools, integration)
- **Hours 3-4**: Implement ReAct pattern (reasoning + acting in agents)
- **Hours 5-6**: Create example tasks (research, coding, analysis workflows)
- **Hour 7**: Build terminal UI (agent activity logging, decision trace)
- **Hour 8**: Documentation and polish (README, examples, cleanup)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-92
  - Days 71-80: LLM fundamentals and LangChain
  - Days 81-92: Advanced GenAI patterns
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 40-53
  - Days 40-45: Agentic AI patterns
  - Days 46-53: Advanced agent architectures

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended)
- Understanding of LLMs and agents
- Basic knowledge of state machines

### Tools Needed
- Python with langgraph, langchain, ollama
- Ollama for local LLM serving
- DuckDuckGo for web search
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install Dependencies
```bash
# Install core dependencies
pip install langgraph langchain langchain-community ollama

# Install tools
pip install duckduckgo-search wikipedia-api gradio

# Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b
ollama pull mistral:7b
```

### Step 3: Define Agent State
```python
# src/state.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    current_agent: str
    results: dict
    next_action: str
    iteration: int
    max_iterations: int

def create_initial_state(task: str) -> AgentState:
    """Create initial state for a task"""
    return {
        "messages": [],
        "task": task,
        "current_agent": "planner",
        "results": {},
        "next_action": "plan",
        "iteration": 0,
        "max_iterations": 10
    }
```

### Step 4: Build Tools
```python
# src/tools/search.py
from langchain.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo"""
    try:
        results = DDGS().text(query, max_results=5)
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}\n   {result['body']}\n   URL: {result['href']}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information"""
    import wikipedia
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

# src/tools/code_exec.py
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

@tool
def execute_python(code: str) -> str:
    """Execute Python code safely and return output"""
    # Create string buffers for stdout and stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    try:
        # Redirect stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Create restricted globals (no dangerous imports)
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "sorted": sorted,
                    "list": list,
                    "dict": dict,
                    "str": str,
                    "int": int,
                    "float": float,
                }
            }
            
            # Execute code
            exec(code, safe_globals)
        
        # Get output
        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()
        
        if errors:
            return f"Errors:\n{errors}\n\nOutput:\n{output}"
        return output if output else "Code executed successfully (no output)"
        
    except Exception as e:
        return f"Execution error: {str(e)}"

# src/tools/file_ops.py
@tool
def read_file(path: str) -> str:
    """Read contents of a file"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file"""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
```

### Step 5: Create Specialized Agents
```python
# src/agents/planner.py
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

class PlannerAgent:
    """Plans and decomposes tasks, routes to appropriate agents"""
    
    def __init__(self, model_name: str = "llama3:8b"):
        self.llm = Ollama(model=model_name, temperature=0.7)
        
        self.prompt = PromptTemplate(
            template="""You are a Planner agent that decomposes tasks and routes them to specialized agents.

Available agents:
- Researcher: Searches web and gathers information
- Coder: Writes and executes Python code
- Analyst: Analyzes data and provides insights

Task: {task}

Current results: {results}

Analyze the task and decide:
1. What needs to be done next?
2. Which agent should handle it?
3. What specific instruction should that agent receive?

Respond in this format:
AGENT: [researcher/coder/analyst/done]
INSTRUCTION: [specific instruction for the agent]
REASONING: [why this agent and action]""",
            input_variables=["task", "results"]
        )
    
    def plan(self, state: dict) -> dict:
        """Plan next action based on current state"""
        prompt = self.prompt.format(
            task=state["task"],
            results=state.get("results", {})
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse response
        lines = response.split("\n")
        agent = "done"
        instruction = ""
        
        for line in lines:
            if line.startswith("AGENT:"):
                agent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("INSTRUCTION:"):
                instruction = line.split(":", 1)[1].strip()
        
        return {
            "next_agent": agent,
            "instruction": instruction,
            "reasoning": response
        }

# src/agents/researcher.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

class ResearcherAgent:
    """Searches web and gathers information"""
    
    def __init__(self, model_name: str = "llama3:8b"):
        from src.tools.search import web_search, wikipedia_search
        
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.tools = [web_search, wikipedia_search]
        
        self.prompt = PromptTemplate(
            template="""You are a Researcher agent. Use the available tools to gather information.

Tools available:
{tools}

Task: {instruction}

Think step by step:
1. What information do I need?
2. Which tool should I use?
3. How should I search?

{agent_scratchpad}""",
            input_variables=["instruction", "tools", "agent_scratchpad"]
        )
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def research(self, instruction: str) -> str:
        """Execute research task"""
        result = self.executor.invoke({"input": instruction})
        return result["output"]

# src/agents/coder.py
class CoderAgent:
    """Writes and executes Python code"""
    
    def __init__(self, model_name: str = "llama3:8b"):
        from src.tools.code_exec import execute_python
        
        self.llm = Ollama(model=model_name, temperature=0.2)
        self.tools = [execute_python]
        
        self.prompt = PromptTemplate(
            template="""You are a Coder agent. Write Python code to solve the task.

Task: {instruction}

Write clean, well-commented Python code. Test your code before finalizing.

Available tools:
- execute_python: Run Python code and see output

Think step by step:
1. What does the code need to do?
2. Write the code
3. Test it
4. Return the result

{agent_scratchpad}""",
            input_variables=["instruction", "agent_scratchpad"]
        )
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def code(self, instruction: str) -> str:
        """Execute coding task"""
        result = self.executor.invoke({"input": instruction})
        return result["output"]

# src/agents/analyst.py
class AnalystAgent:
    """Analyzes data and provides insights"""
    
    def __init__(self, model_name: str = "llama3:8b"):
        from src.tools.code_exec import execute_python
        from src.tools.file_ops import read_file
        
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.tools = [execute_python, read_file]
        
        self.prompt = PromptTemplate(
            template="""You are an Analyst agent. Analyze data and provide insights.

Task: {instruction}

Available data: {data}

Think step by step:
1. What analysis is needed?
2. What patterns or insights can I find?
3. What conclusions can I draw?

{agent_scratchpad}""",
            input_variables=["instruction", "data", "agent_scratchpad"]
        )
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def analyze(self, instruction: str, data: str = "") -> str:
        """Execute analysis task"""
        result = self.executor.invoke({
            "input": instruction,
            "data": data
        })
        return result["output"]
```

### Step 6: Implement ReAct Pattern

The ReAct (Reasoning + Acting) pattern makes agents think before acting:

```python
# src/agents/researcher.py - Enhanced with ReAct
class ResearcherAgent:
    """Searches web using ReAct pattern"""
    
    def __init__(self, model_name: str = "llama3:8b"):
        from src.tools.search import web_search, wikipedia_search
        
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.tools = [web_search, wikipedia_search]
        
        # ReAct prompt template
        self.prompt = PromptTemplate(
            template="""You are a Researcher agent using the ReAct pattern.

Task: {instruction}

Think step by step using this format:

Thought: What do I need to find out?
Action: [tool_name]
Action Input: [input to tool]
Observation: [tool result]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: [comprehensive answer]

Available tools:
{tools}

Begin!

{agent_scratchpad}""",
            input_variables=["instruction", "tools", "agent_scratchpad"]
        )
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def research(self, instruction: str) -> str:
        """Execute research with ReAct reasoning"""
        result = self.executor.invoke({"input": instruction})
        return result["output"]
```

### Step 7: Build LangGraph State Machine
```python
# src/graph.py
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.coder import CoderAgent
from src.agents.analyst import AnalystAgent

class MultiAgentSystem:
    """Orchestrates multiple agents using LangGraph"""
    
    def __init__(self):
        # Initialize agents
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.coder = CoderAgent()
        self.analyst = AnalystAgent()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("coder", self.coder_node)
        workflow.add_node("analyst", self.analyst_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edges from planner
        workflow.add_conditional_edges(
            "planner",
            self.route_to_agent,
            {
                "researcher": "researcher",
                "coder": "coder",
                "analyst": "analyst",
                "done": END
            }
        )
        
        # All agents return to planner
        workflow.add_edge("researcher", "planner")
        workflow.add_edge("coder", "planner")
        workflow.add_edge("analyst", "planner")
        
        return workflow.compile()
    
    def planner_node(self, state: AgentState) -> AgentState:
        """Planner agent node"""
        print(f"\n🧠 Planner (Iteration {state['iteration']})")
        
        # Check max iterations
        if state["iteration"] >= state["max_iterations"]:
            state["next_action"] = "done"
            return state
        
        # Plan next action
        plan = self.planner.plan(state)
        
        state["next_action"] = plan["next_agent"]
        state["current_instruction"] = plan["instruction"]
        state["iteration"] += 1
        
        print(f"   Next: {plan['next_agent']}")
        print(f"   Instruction: {plan['instruction'][:100]}...")
        
        return state
    
    def researcher_node(self, state: AgentState) -> AgentState:
        """Researcher agent node"""
        print(f"\n🔍 Researcher")
        
        result = self.researcher.research(state["current_instruction"])
        state["results"]["researcher"] = result
        
        print(f"   Result: {result[:200]}...")
        return state
    
    def coder_node(self, state: AgentState) -> AgentState:
        """Coder agent node"""
        print(f"\n💻 Coder")
        
        result = self.coder.code(state["current_instruction"])
        state["results"]["coder"] = result
        
        print(f"   Result: {result[:200]}...")
        return state
    
    def analyst_node(self, state: AgentState) -> AgentState:
        """Analyst agent node"""
        print(f"\n📊 Analyst")
        
        data = state["results"].get("researcher", "") + state["results"].get("coder", "")
        result = self.analyst.analyze(state["current_instruction"], data)
        state["results"]["analyst"] = result
        
        print(f"   Result: {result[:200]}...")
        return state
    
    def route_to_agent(self, state: AgentState) -> str:
        """Route to next agent based on planner decision"""
        return state["next_action"]
    
    def run(self, task: str) -> dict:
        """Run the multi-agent system on a task"""
        from src.state import create_initial_state
        
        print(f"\n{'='*80}")
        print(f"TASK: {task}")
        print(f"{'='*80}")
        
        initial_state = create_initial_state(task)
        final_state = self.graph.invoke(initial_state)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS:")
        print(f"{'='*80}")
        for agent, result in final_state["results"].items():
            print(f"\n{agent.upper()}:")
            print(result)
        
        return final_state

# Usage
if __name__ == "__main__":
    system = MultiAgentSystem()
    
    # Example tasks
    tasks = [
        "Research the latest developments in quantum computing and summarize the key findings",
        "Write Python code to calculate the Fibonacci sequence and test it",
        "Analyze the trend of AI adoption in healthcare over the past 5 years"
    ]
    
    for task in tasks:
        result = system.run(task)
```

### Step 8: Create Terminal UI for Monitoring

```python
# app.py - Terminal UI with agent activity logging
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from src.graph import MultiAgentSystem
import time

console = Console()

class TerminalUI:
    """Real-time terminal UI for agent monitoring"""
    
    def __init__(self):
        self.system = MultiAgentSystem()
        self.agent_logs = []
        self.current_agent = None
    
    def create_display(self):
        """Create rich display table"""
        table = Table(title="Multi-Agent System Monitor")
        
        table.add_column("Time", style="cyan")
        table.add_column("Agent", style="magenta")
        table.add_column("Action", style="green")
        table.add_column("Status", style="yellow")
        
        for log in self.agent_logs[-10:]:  # Show last 10
            table.add_row(
                log["time"],
                log["agent"],
                log["action"][:50],
                log["status"]
            )
        
        return Panel(table, title=f"Current: {self.current_agent or 'Idle'}")
    
    def log_agent_action(self, agent, action, status="running"):
        """Log agent activity"""
        self.agent_logs.append({
            "time": time.strftime("%H:%M:%S"),
            "agent": agent,
            "action": action,
            "status": status
        })
        self.current_agent = agent
    
    def run_task(self, task: str):
        """Run task with live monitoring"""
        console.print(f"\n[bold]Task:[/bold] {task}\n")
        
        with Live(self.create_display(), refresh_per_second=2) as live:
            # Hook into system to log actions
            result = self.system.run(task)
            
            live.update(self.create_display())
        
        console.print("\n[bold green]✓ Task Complete![/bold green]\n")
        return result

# Usage
if __name__ == "__main__":
    ui = TerminalUI()
    
    tasks = [
        "Research quantum computing and write Python code to simulate a qubit",
        "Find climate change data and analyze the trends",
    ]
    
    for task in tasks:
        ui.run_task(task)
```

### Step 9: Create Gradio Interface (Optional)
```python
# app.py
import gradio as gr
from src.graph import MultiAgentSystem

# Initialize system
print("Initializing Multi-Agent System...")
system = MultiAgentSystem()
print("✓ System ready!")

def process_task(task, max_iterations):
    """Process task through multi-agent system"""
    try:
        # Update max iterations
        from src.state import create_initial_state
        initial_state = create_initial_state(task)
        initial_state["max_iterations"] = max_iterations
        
        # Run system
        final_state = system.graph.invoke(initial_state)
        
        # Format results
        output = f"**Task:** {task}\n\n"
        output += f"**Iterations:** {final_state['iteration']}\n\n"
        output += "**Results:**\n\n"
        
        for agent, result in final_state["results"].items():
            output += f"### {agent.upper()}\n{result}\n\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=process_task,
    inputs=[
        gr.Textbox(
            label="Task",
            placeholder="Enter a complex task that requires multiple agents...",
            lines=3
        ),
        gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Max Iterations"
        )
    ],
    outputs=gr.Markdown(label="Results"),
    title="🤖 Multi-Agent AI System",
    description="A system where specialized agents (Planner, Researcher, Coder, Analyst) work together to solve complex tasks.",
    examples=[
        ["Research quantum computing and write Python code to simulate a qubit", 10],
        ["Find information about climate change and analyze the data trends", 8],
        ["Research machine learning algorithms and implement a simple classifier", 12]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False)
```

## Key Features to Implement

### 1. Specialized Agents
- **Planner Agent**: Task decomposition and planning, routes to other agents
- **Researcher Agent**: Web search and information gathering (DuckDuckGo, Wikipedia)
- **Coder Agent**: Write and review code, execute Python safely
- **Analyst Agent**: Data analysis and insights, visualization

### 2. Tool Ecosystem
- **Web search**: DuckDuckGo API for real-time information
- **Code execution**: Sandboxed Python REPL
- **File operations**: Read/write files safely
- **Calculator**: Mathematical computations
- **Custom tools**: Extensible tool framework

### 3. Agent Communication
- **Message passing**: Via LangGraph state
- **Shared state management**: Persistent across agents
- **Agent handoffs**: Smooth transitions between agents
- **Conversation history**: Full context maintained

### 4. Planning & Reasoning
- **Task decomposition**: Break complex tasks into steps
- **ReAct pattern**: Reasoning + Acting cycle
- **Chain-of-thought prompting**: Explicit reasoning steps
- **Self-reflection**: Agents evaluate their own outputs

### 5. Orchestration (LangGraph)
- **State machine**: Nodes for each agent
- **Conditional routing**: Dynamic agent selection
- **Parallel execution**: Multiple agents simultaneously (optional)
- **Error handling**: Graceful failure recovery

### 6. Monitoring
- **Agent activity logging**: Track all agent actions
- **Decision trace visualization**: See reasoning process
- **Performance metrics**: Time, tool calls, success rate
- **Terminal UI**: Real-time agent interactions display

## Success Criteria

By the end of this project, you should have:

### Functionality
- [ ] 4 specialized agents implemented (Planner, Researcher, Coder, Analyst)
- [ ] Tool ecosystem working (web search, code exec, file ops, calculator)
- [ ] LangGraph state machine orchestrating agents
- [ ] Agent communication via shared state
- [ ] ReAct pattern implemented in agents
- [ ] Complex tasks completed successfully
- [ ] Terminal UI showing agent interactions

### Quality Metrics
- [ ] **Agents collaborate successfully**: Multi-step tasks complete
- [ ] **Tools used appropriately**: No redundant calls
- [ ] **Complex tasks completed**: Research + code + analysis workflows
- [ ] **Code quality**: < 700 lines of code
- [ ] **Reasonable response times**: < 5 minutes for complex tasks
- [ ] **Efficient tool use**: Minimal redundant API calls

### Deliverables
- [ ] 4 specialized agents with clear roles
- [ ] Tool ecosystem (5+ tools)
- [ ] LangGraph orchestration with conditional routing
- [ ] Example workflows (3+ complex tasks)
- [ ] Terminal UI with agent activity logging
- [ ] Decision trace visualization
- [ ] Comprehensive documentation
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Design multi-agent architectures
- Implement LangGraph state machines
- Coordinate multiple specialized agents
- Build tool-calling agents
- Manage agent state and memory
- Handle agent communication
- Debug complex agent workflows
- Explain multi-agent patterns

## Expected Performance

**Task Completion**:
```
Simple tasks (1 agent): 1-2 iterations, 30-60 seconds
Medium tasks (2-3 agents): 3-5 iterations, 2-4 minutes
Complex tasks (all agents): 5-10 iterations, 5-10 minutes
```

**Agent Coordination**:
```
Planner → Researcher → Planner → Coder → Planner → Analyst → Done
Average: 6-8 agent calls per complex task
```

## Project Structure

```
project-c-multi-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py       # Task decomposition & routing
│   │   ├── researcher.py    # Web search & info gathering
│   │   ├── coder.py         # Code writing & execution
│   │   └── analyst.py       # Data analysis & insights
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py        # DuckDuckGo & Wikipedia
│   │   ├── code_exec.py     # Sandboxed Python REPL
│   │   ├── file_ops.py      # File read/write
│   │   └── calculator.py    # Math operations
│   ├── graph.py             # LangGraph state machine
│   ├── state.py             # Shared state management
│   └── monitoring.py        # Logging & visualization
├── examples/
│   ├── research_task.py     # Web research workflow
│   ├── coding_task.py       # Code generation workflow
│   ├── analysis_task.py     # Data analysis workflow
│   └── complex_task.py      # Multi-agent collaboration
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_graph.py
├── results/
│   ├── agent_traces/        # Decision logs
│   └── performance_metrics.json
├── app.py                   # Terminal UI
├── requirements.txt
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Infinite Loops
**Problem**: Agents keep calling each other without completing
**Solution**: Implement max_iterations limit, add "done" condition in planner

### Challenge 2: Tool Execution Failures
**Problem**: Tools fail or return errors
**Solution**: Add try-except blocks, return error messages to agent for retry

### Challenge 3: Agent Confusion
**Problem**: Agents don't understand which agent to call next
**Solution**: Improve planner prompt with clear examples, add reasoning step

### Challenge 4: State Management
**Problem**: State gets corrupted or lost between agents
**Solution**: Use TypedDict for state validation, log state changes

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with workflow diagrams
2. **Write Blog Post**: "Building Multi-Agent Systems with LangGraph"
3. **Extend Features**: Add more agents (Writer, Critic, Validator)
4. **Build Project D**: Continue with Computer Vision Pipeline
5. **Production Use**: Deploy with FastAPI backend

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Multi-Agent Patterns](https://www.anthropic.com/research/building-effective-agents)
- [Agent Architectures](https://lilianweng.github.io/posts/2023-06-23-agent/)

## Questions?

If you get stuck:
1. Review the tech-spec.md for detailed agent implementations
2. Check LangGraph documentation for state machine patterns
3. Search LangChain community forums
4. Review the 100 Days bootcamp materials on agentic AI
