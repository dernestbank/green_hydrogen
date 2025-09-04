import os
import requests
from typing import Optional, List, Dict

class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"
        self.headers = {"Content-Type": "application/json"}

    def search(self, query: str, topic: str = "general", time_range: Optional[str] = None,
               max_results: int = 10, search_depth: str = "basic") -> Dict:
        """
        Search using Tavily API
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "search_depth": search_depth
        }

        if time_range:
            payload["time_range"] = time_range

        try:
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

# Test the API
if __name__ == "__main__":
    # Your API key
    api_key = "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"

    client = TavilyClient(api_key)

    print("Testing Tavily API with hydrogen production news...")
    print("=" * 60)

    result = client.search(
        query="latest news hydrogen production",
        topic="news",
        time_range="week",
        max_results=5,
        search_depth="basic"
    )

    if "error" in result:
        print(f"API Error: {result['error']}")
    else:
        print("API call successful!")
        print(f"Total results: {len(result.get('results', []))}")
        print("\nTop headlines:")
        print("-" * 40)

        for i, article in enumerate(result.get('results', [])[:5], 1):
            print(f"{i}. {article.get('title', 'No title')}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   URL: {article.get('url', 'No URL')}")
            print(f"   Published: {article.get('published_date', 'No date')}")
            print()

        if result.get('answer'):
            print("AI Summary:")
            print("-" * 40)
            print(result['answer'])