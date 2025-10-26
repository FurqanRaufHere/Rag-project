from typing import List, Dict
import textwrap

SYSTEM_MESSAGE = (
    "You are an assistant that answers questions using only the provided CONTEXT from a document. "
    "Cite the page number(s) where you found the information in parentheses like (page 12). "
    "If the answer cannot be found in the context, reply: 'Answer not found in the document.' "
    "Be concise and avoid hallucinations."
)

def build_rag_messages(retrieved_chunks: List[Dict], question: str, max_context_chars=3000):
    """
    Return messages list for chat-based API (role/content dicts).
    Trims context if necessary to avoid extremely long prompts.
    """
    # sort chunks by score descending (already likely)
    chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)

    # build context string with page label
    ctx_parts = []
    total_len = 0
    for c in chunks:
        part = f"Page {c.get('page', '?')}:\n{c['text']}\n---\n"
        total_len += len(part)
        if total_len > max_context_chars:
            break
        ctx_parts.append(part)

    context = "\n".join(ctx_parts).strip()

    user_content = f"""Context:
{context}

Question: {question}
"""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]
    return messages
