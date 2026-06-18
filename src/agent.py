from typing import Annotated
from typing_extensions import TypedDict
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import expense_search, expense_calculator, expense_comparator
from prompts import SYSTEM_PROMPT



# state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# LLM  + tools
tools_list = [expense_search, expense_calculator, expense_comparator]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools_list)



# nodes
def chatbot_node(state: AgentState) -> AgentState:
    """LLM turn - reads messages, either replies or emits a tool call."""
    
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Route to tools if the LLM emitted tool calls, otherwise end."""
    
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    
    return END

tool_node = ToolNode(tools_list)



# graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools",   tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue)
graph_builder.add_edge("tools", "chatbot")

agent = graph_builder.compile()



# chat function
def chat(user_input: str, chat_history: list) -> tuple[str, list]:
    """
    Run one turn. Pass chat_history in, get (reply, updated_history) back.
    """
    new_message = HumanMessage(content=user_input)
    state = agent.invoke({"messages": chat_history + [new_message]})
    all_messages    = state["messages"]
    assistant_reply = all_messages[-1].content
    return assistant_reply, all_messages



# CLI loop
if __name__ == "__main__":
    history = []
    print("💬 Expense Chatbot  —  type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        answer, history = chat(q, history)
        print(f"\nAssistant: {answer}\n")