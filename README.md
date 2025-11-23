This project is an AI-powered anime recommendation and question-answering API built using Gemini 2.5 Flash, Chroma Vector DB, LangChain, and HuggingFace embeddings. It semantically searches an anime dataset and generates meaningful responses, summaries, and suggestions based on user queries.

✅ Key Features

📚 Loads 2023 anime dataset (Name, Synopsis, Genres, Ratings, etc.)

🧠 Converts text into embeddings using all-MiniLM-L6-v2

🔍 Stores vectors in Chroma DB for fast semantic search

🤖 Uses Gemini to generate context-aware recommendations & answers

🚀 Exposed as a FastAPI endpoint for frontend or chatbot integration

🛠️ Tech Stack

Python, FastAPI

LangChain, Chroma, HuggingFace Embeddings

Gemini 2.5 Flash (Google Generative AI)

Pandas, Pydantic

💡 Use Cases

Anime suggestion chatbot

Story/genre/character lookup

Anime similarity search

Recommendation engines for streaming apps

📂 Project Flow

CSV → Clean & chunk dataset → Create embeddings → Store in Chroma → Retrieve top-k matches → Gemini generates final recommendation
