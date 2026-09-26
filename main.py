# main.py
import sys
from graph import create_agent_graph, llm
from rag import retrieve_relevant_chunks

def format_briefing(briefing: dict):
    print("\n" + "="*80)
    print(f"📑 EXECUTIVE BRIEFING: {briefing['title']}")
    print("="*80)
    print(f"• Authors:     {', '.join(briefing['authors'])}")
    print(f"• arXiv ID:    {briefing['arxiv_id']} | Date: {briefing['publish_date']}")
    print(f"• Link:        {briefing['paper_link']}")
    print("\n💡 WHY THIS MATTERS (Summary):")
    print(f"{briefing['one_paragraph_summary']}")
    print("\n🎯 PROBLEM STATEMENT:")
    print(f"{briefing['problem_statement']}")
    print("\n🛠️ METHOD / APPROACH:")
    for pt in briefing['method_approach']:
        print(f"  - {pt}")
    print("\n📊 KEY RESULTS / CLAIMS:")
    for pt in briefing['key_results']:
        print(f"  - {pt}")
    print("\n⚠️ EXPLICIT LIMITATIONS:")
    for pt in briefing['limitations']:
        print(f"  - {pt}")
    print("\n❓ SUGGESTED FOLLOW-UP QUESTIONS:")
    for q in briefing['suggested_followups']:
        print(f"  ? {q}")
    print("="*80 + "\n")

def run_qa_session(collection, paper_title: str):
    print(f"\n💬 Entering QA Mode for '{paper_title}'.")
    print("Ask any follow-up question (type 'exit' or 'q' to return):\n")
    
    qa_history = []
    
    while True:
        try:
            user_question = input("User >> ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "q", "quit"]:
                print("Exiting QA session. Goodbye!")
                break
                
            # 1. Retrieve grounded context
            chunks = retrieve_relevant_chunks(collection, user_question, top_k=4)
            context = "\n---\n".join(chunks)
            
            # 2. Strict anti-hallucination grounding prompt
            prompt = f"""You are an academic researcher answering questions strictly based on the paper excerpt provided below.
If the answer is NOT explicitly mentioned or cannot be verified directly from the excerpt, say:
"The paper does not provide enough evidence or mention this detail."
Do NOT fabricate or extrapolate information.

Excerpt from Paper:
\"\"\"
{context}
\"\"\"

Question: {user_question}
Answer:"""

            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            print(f"\nAgent >> {answer}\n")
            qa_history.append({"question": user_question, "answer": answer})
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting QA session.")
            break

def main():
    print("="*60)
    print(" Autonomous arXiv Paper Digest & QA Agent")
    print("="*60)
    
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter an arXiv ID, URL, or Research Topic: ").strip()
        
    if not user_query:
        print("Error: Input cannot be empty.")
        return

    print("\n⏳ Processing state graph pipeline...")
    agent_graph = create_agent_graph()
    
    initial_state = {
        "user_input": user_query,
        "query_type": "",
        "arxiv_id": None,
        "paper_metadata": None,
        "candidate_papers": [],
        "pdf_path": None,
        "extracted_text": "",
        "parse_fallback": False,
        "vector_store": None,
        "briefing": None,
        "qa_history": [],
        "error": None
    }
    
    final_state = agent_graph.invoke(initial_state)
    
    if final_state.get("error"):
        print(f"\n❌ Error encountered: {final_state['error']}")
        return
        
    # Display the executive briefing
    format_briefing(final_state["briefing"])
    
    if final_state.get("parse_fallback"):
        print("⚠️ Note: Full PDF text could not be extracted; briefing was generated from the abstract.")

    # Start follow-up QA Loop
    if final_state.get("vector_store"):
        run_qa_session(final_state["vector_store"], final_state["paper_metadata"]["title"])

if __name__ == "__main__":
    main()