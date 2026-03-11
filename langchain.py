from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
import os
import time
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

print("Loaded key:", os.getenv("OPENAI_API_KEY"))

file_path = Path(input("Enter the file path "))

def input_path(file_path):
    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    elif file_path.suffix.lower() == ".docx":
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are allowed.")
    return docs

docs = input_path(file_path)

client = OpenAI()

queries = [
    "what was the revenue in Q3 2024",
    "what was the revenue in Q3 2023",
    "what was the customer satisfaction score",
    "what market share does North America have",
    "what was the production capacity utilization"
]

answers = {
    "what was the revenue in Q3 2024": "125.3",
    "what was the revenue in Q3 2023": "108.8",
    "what was the customer satisfaction score": "4.6",
    "what market share does North America have": "45%",
    "what was the production capacity utilization": "87%"
}

# ---------------- CHARACTER ----------------

splitter = CharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
chunks = splitter.split_documents(docs)

Embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [chunk.page_content for chunk in chunks]
values = Embeddings.embed_documents(texts)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=Embeddings
)

vectorstore.save_local("faiss_index2")
vectorstore = FAISS.load_local(
    "faiss_index2",
    Embeddings,
    allow_dangerous_deserialization=True
)

correct = 0
total_latency = 0

for query in queries:
    start = time.time()
    results = vectorstore.similarity_search(query, k=3)
    end = time.time()

    latency = end - start
    total_latency += latency

    print("\nQuery:", query)
    print("latency", latency, "seconds")

    for i, doc in enumerate(results, start=1):
        print("\nResult", i)
        print(doc.page_content[:500])

    retrieved_text = " ".join([doc.page_content for doc in results])

    if answers[query].lower() in retrieved_text.lower():
        correct += 1

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Based only on the text below, answer the question.

Question:
{query}

Text:
{context}

Return only the short final answer.
No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("LLM answer:", response.choices[0].message.content)

accuracy = correct / len(queries)
avg_latency = total_latency / len(queries)

print("chunks:", len(chunks))
print("\nCharacter splitter accuracy:", accuracy)
print("Character splitter average latency:", avg_latency)


# ---------------- RECURSIVE ----------------

splitter = RecursiveCharacterTextSplitter(chunk_size = 300 ,chunk_overlap = 50)
recchunks = splitter.split_documents(docs)

Embedding = OpenAIEmbeddings(model="text-embedding-3-small")

textx = [chunk.page_content for chunk in recchunks]
values = Embedding.embed_documents(textx)

FAISS.from_documents(recchunks, Embedding).save_local("faiss_index3")

vectorstore = FAISS.load_local(
    "faiss_index3",
    Embedding,
    allow_dangerous_deserialization=True
)

correct = 0
total_latency = 0

for query in queries:
    start = time.time()
    results = vectorstore.similarity_search(query, k=3)
    end = time.time()

    latency = end - start
    total_latency += latency

    print("\nQuery:", query)
    print("latency", latency, "seconds")

    for i, doc in enumerate(results, start=1):
        print("\nResult", i)
        print(doc.page_content[:500])

    retrieved_text = " ".join([doc.page_content for doc in results])

    if answers[query].lower() in retrieved_text.lower():
        correct += 1

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Based only on the text below, answer the question.

Question:
{query}

Text:
{context}

Return only the short final answer.
No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("LLM answer:", response.choices[0].message.content)

accuracy = correct / len(queries)
avg_latency = total_latency / len(queries)

print("chunks:", len(recchunks))
print("\nRecursive splitter accuracy:", accuracy)
print("Recursive splitter average latency:", avg_latency)


# ---------------- TOKEN ----------------

splitter = TokenTextSplitter(chunk_size = 200, chunk_overlap  = 30)
textchunks = splitter.split_documents(docs)

Embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [chunk.page_content for chunk in textchunks]
values = Embeddings.embed_documents(texts)

FAISS.from_documents(textchunks, Embeddings).save_local("faiss_index4")

vectorstore = FAISS.load_local(
    "faiss_index4",
    Embeddings,
    allow_dangerous_deserialization=True
)

correct = 0
total_latency = 0

for query in queries:
    start = time.time()
    results = vectorstore.similarity_search(query, k=3)
    end = time.time()

    latency = end - start
    total_latency += latency

    print("\nQuery:", query)
    print("latency", latency, "seconds")

    for i, doc in enumerate(results, start=1):
        print("\nResult", i)
        print(doc.page_content[:500])

    retrieved_text = " ".join([doc.page_content for doc in results])

    if answers[query].lower() in retrieved_text.lower():
        correct += 1

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Based only on the text below, answer the question.

Question:
{query}

Text:
{context}

Return only the short final answer.
No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("LLM answer:", response.choices[0].message.content)

accuracy = correct / len(queries)
avg_latency = total_latency / len(queries)
print("chunks:", len(chunks))
print("\nToken splitter accuracy:", accuracy)
print("Token splitter average latency:", avg_latency)




    