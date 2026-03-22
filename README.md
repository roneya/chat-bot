FAQ Chatbot
A RAG-based customer support chatbot built with ChromaDB and Mistral AI. Upload your FAQ data, and the bot answers user questions using only your knowledge base — no hallucinations.
What it does

Accepts questions from users via a chat UI
Retrieves relevant answers from a ChromaDB vector store
Generates grounded responses using Mistral AI
Falls back gracefully when no match is found
Lets admins teach the bot new answers directly from the dashboard

Tech Stack

Backend — Python, Flask
Vector DB — ChromaDB (SQLite-backed)
LLM — Mistral AI
Analytics — SQLite
Deployment — Render (Docker)

Admin Dashboard

Upload CSV files to bulk-load Q&A pairs into ChromaDB
View all knowledge base entries, add or delete them
See unanswered queries and add answers on the spot
Track users, queries, response times, and relevance scores

Knowledge Base
The bot reads exclusively from ChromaDB — CSV files are only used for bulk ingestion. Admins can also add individual Q&A pairs directly from the dashboard without touching any files.
