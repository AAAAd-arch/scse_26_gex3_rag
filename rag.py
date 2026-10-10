### COMPLETE THE CODE ###

from ollama import chat
from policy_loader import load_policy_documents
from rag_tool import retrieve_documents


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


def build_context(results):
    context = ""

    for document in results:
        context += (
            "Policy ID: " + document.metadata["policy_id"] + "\n"
            "Topic: " + document.metadata["topic"] + "\n"
            + document.page_content + "\n\n"
        )

    return context


def answer_question(question):
    results = retrieve_documents(question)
    context = build_context(results)

    system_prompt = """You are a university IT support assistant.
Answer the user's question using only the retrieved university IT policy context.
Do not invent or guess information.
If the requested information is not explicitly provided in the context, clearly say that it is not available in the university IT policies.
Do not add unrelated advice or information.
Keep the answer short and direct."""

    user_prompt = f"""Retrieved university IT policy context:

{context}
Question:
{question}"""

    response = chat(
        model="qwen3:0.6b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.message.content