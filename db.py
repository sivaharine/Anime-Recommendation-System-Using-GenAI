


import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1️⃣ Load dataset
df = pd.read_csv("data/anime-dataset-2023.csv")  # replace with your actual CSV name
#removing file witout synopsis
df = df.dropna(subset=["Synopsis", "Name"])

print(f"✅ Dataset ready for embedding: {df.shape}")

# 2️⃣ Prepare texts and metadata
texts = df["Synopsis"].tolist()
metadatas = [
    {
        "name": row["Name"],
        "english_name": row.get("English name", ""),
        "genres": row.get("Genres", ""),
        "type": row.get("Type", ""),
        "rating": row.get("Rating", ""),
        "image_url": row.get("Image URL", ""),
    }
    for _, row in df.iterrows()
]

# 3️⃣ Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4️⃣ Create and store Chroma vector DB
vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="chroma_db"
)

# ✅ Newer versions auto-persist, no need for .persist() or _client.persist()
print("✅ Chroma DB successfully created and stored in 'chroma_db' folder!")
