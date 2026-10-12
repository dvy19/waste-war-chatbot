
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class TavilySearch:

    def __init__(self):
        self.client = TavilyClient(
            api_key=os.getenv("TAVILY_API_KEY")
        )

    def search(self, query: str) -> list[str]:

        response = self.client.search(
            query=query,
            max_results=3,
            search_depth="basic"
        )

        results = response.get("results", [])

        return [
            f"Title: {item['title']}\n"
            f"Content: {item['content']}\n"
            f"URL: {item['url']}"
            for item in results
        ]