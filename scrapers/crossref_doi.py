import requests
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class CrossrefDOIFetcher:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        self.headers = {
            "User-Agent": os.getenv("CR_API_AGENT", "CrossrefDOIFetcher/1.0"),
            "Mailto": os.getenv("CR_API_MAILTO")
        }
        if os.getenv("CR_API_PLUS"):
            self.headers["Crossref-Plus-API-Token"] = os.getenv("CR_API_PLUS")

    def get_author_dois(self, author_name: str, max_results: int = 100) -> List[str]:
        """
        Retrieve DOIs for works by the specified author.
        
        :param author_name: Name of the author to search for
        :param max_results: Maximum number of results to return
        :return: List of DOIs
        """
        params = {
            "query.author": f"{author_name}",
            "rows": min(max_results, 1000),  # Crossref API limit is 1000
            "select": "DOI,title",
        }

        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            dois = []
            for item in data.get("message", {}).get("items", []):
                doi = item.get("DOI")
                if doi:
                    dois.append(doi)

            return dois[:max_results]

        except requests.RequestException as e:
            print(f"Error fetching DOIs for {author_name}: {e}")
            return []

    def get_paper_details(self, doi: str) -> Dict:
        """
        Retrieve details for a specific paper given its DOI.
        
        :param doi: DOI of the paper
        :return: Dictionary containing paper details
        """
        try:
            response = requests.get(f"{self.base_url}/{doi}", headers=self.headers)
            response.raise_for_status()
            data = response.json()

            item = data.get("message", {})
            return {
                "title": item.get("title", [None])[0],
                "DOI": item.get("DOI"),
                "URL": item.get("URL"),
                "type": item.get("type"),
                "published": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
            }

        except requests.RequestException as e:
            print(f"Error fetching details for DOI {doi}: {e}")
            return {}

if __name__ == "__main__":
    fetcher = CrossrefDOIFetcher()
    
    author_name = "William Merrill"
    dois = fetcher.get_author_dois(author_name, max_results=5)
    
    print(f"DOIs for papers by {author_name}:")
    for doi in dois:
        print(doi)
        details = fetcher.get_paper_details(doi)
        print(f"Title: {details.get('title')}")
        print(f"Type: {details.get('type')}")
        print(f"Published: {details.get('published')}")
        print(f"URL: {details.get('URL')}")
        print()
