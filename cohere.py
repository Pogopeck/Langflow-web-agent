from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

load_dotenv()

# Initialize LLM
llm = init_chat_model("command-a-03-2025")

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None

# Dummy search functions
def serp_search(query: str, engine: str = "google") -> str:
    return f"[{engine.upper()}] Search results for: '{query}'"

def reddit_search_api(keyword: str) -> str:
    return f"[REDDIT] Posts related to: '{keyword}'"

# Node functions
def google_search(state: State):
    print("🔍 Google Search Node")
    query = state.get("user_question", "")
    return {"google_results": serp_search(query, "google")}

def bing_search(state: State):
    print("🔍 Bing Search Node")
    query = state.get("user_question", "")
    return {"bing_results": serp_search(query, "bing")}

def reddit_search(state: State):
    print("🔍 Reddit Search Node")
    query = state.get("user_question", "")
    return {"reddit_results": reddit_search_api(query)}

def analyze_reddit_posts(state: State):
    print("🧠 Analyze Reddit Posts Node")
    return {"selected_reddit_urls": ["https://reddit.com/example1", "https://reddit.com/example2"]}

def retrieve_reddit_posts(state: State):
    print("📥 Retrieve Reddit Posts Node")
    return {"reddit_post_data": ["Post 1 content", "Post 2 content"]}

def analyze_google_results(state: State):
    print("📊 Analyze Google Results Node")
    content = state.get("google_results", "")
    analysis = llm.invoke(f"Analyze the following Google search results:\n{content}")
    return {"google_analysis": analysis}

def analyze_bing_results(state: State):
    print("📊 Analyze Bing Results Node")
    content = state.get("bing_results", "")
    analysis = llm.invoke(f"Analyze the following Bing search results:\n{content}")
    return {"bing_analysis": analysis}

def analyze_reddit_results(state: State):
    print("📊 Analyze Reddit Results Node")
    posts = state.get("reddit_post_data", [])
    combined = "\n".join(posts)
    analysis = llm.invoke(f"Analyze the following Reddit posts:\n{combined}")
    return {"reddit_analysis": analysis}

def synthesize_analyses(state: State):
    print("🧪 Synthesizing Final Answer Node")
    google = state.get("google_analysis", "")
    bing = state.get("bing_analysis", "")
    reddit = state.get("reddit_analysis", "")
    combined = f"Google:\n{google}\n\nBing:\n{bing}\n\nReddit:\n{reddit}"
    final = llm.invoke(f"Synthesize the following analyses into a final answer:\n{combined}")
    return {"final_answer": final}

# Build graph
graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")

graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")
graph_builder.add_edge("reddit_search", "analyze_reddit_posts")
graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")

graph_builder.add_edge("analyze_google_results", "synthesize_analyses")
graph_builder.add_edge("analyze_bing_results", "synthesize_analyses")
graph_builder.add_edge("analyze_reddit_results", "synthesize_analyses")

graph_builder.add_edge("synthesize_analyses", END)

graph = graph_builder.compile()

# Chatbot loop
def run_chatbot():
    print("🤖 Multi-Source Research Agent")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Ask me anything: ")
        if user_input.lower() == "exit":
            print("👋 Bye")
            break

        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None,
            "final_answer": None,
        }

        print("\n🚀 Starting parallel research process...")
        final_state = graph.invoke(state)

        if final_state.get("final_answer"):
            print(f"\n✅ Final Answer:\n{final_state.get('final_answer')}\n")

        print("-" * 80)

# Entry point
if __name__ == "__main__":
    run_chatbot()
