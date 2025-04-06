import os
import google.generativeai as genai
from langchain.embeddings.base import Embeddings
from typing import List
import dotenv

dotenv.load_dotenv()


class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "gemini-embedding-001"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text, task_type="retrieval_document") for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, task_type="retrieval_query")

    def _embed(self, text: str, task_type: str) -> List[float]:
        response = genai.embed_content(
            model=self.model_name, content=text, task_type=task_type, title=""
        )
        return response["embedding"]
