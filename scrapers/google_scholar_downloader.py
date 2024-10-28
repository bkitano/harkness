import requests
from typing import List, Dict
from time import sleep
import random
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class GoogleScholarDownloader:
    def __init__(self, language: str = "en"):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is not set")
        self.base_url = "https://serpapi.com/search.json"
        self.output_dir = Path("downloaded_papers")
        self.output_dir.mkdir(exist_ok=True)
        self.language = language

    def search_author(self, author_name: str) -> List[Dict[str, str]]:
        """
        Search Google Scholar for papers by the given author using SerpApi.
        """
        print(f"Searching Google Scholar for papers by {author_name}...")

        params = {
            "engine": "google_scholar",
            "q": f'author:"{author_name}"',
            "api_key": self.api_key,
            "num": 20,  # Adjust as needed, max is 20 per request
            "hl": self.language,  # Set the language parameter
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        papers = []
        for result in data.get("organic_results", []):
            title = result.get("title")
            if result.get("resources"):
                for resource in result.get("resources"):
                    if resource.get("file_format") == "pdf":
                        pdf_link = resource.get("link")
                        papers.append({"title": title, "url": pdf_link})
            else:
                link = result.get("link")
                if title and link:
                    papers.append({"title": title, "url": link})

        return papers

    def download_paper(self, paper: Dict[str, str]) -> None:
        """
        Download a single paper if it's a PDF.
        """
        print(f"Checking: {paper['title']}")

        try:
            response = requests.head(paper['url'], allow_redirects=True)
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type:
                print(f"Downloading PDF: {paper['title']}")
                pdf_response = requests.get(paper['url'], stream=True)
                pdf_response.raise_for_status()

                filename = self._sanitize_filename(paper['title']) + '.pdf'
                filepath = self.output_dir / filename

                with open(filepath, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded: {filename}")
            else:
                print(f"Not a PDF, skipping: {paper['title']}")

        except requests.RequestException as e:
            print(f"Error downloading {paper['title']}: {e}")

        # Delay to be respectful to the server
        sleep(random.uniform(1, 3))

    def _sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        return "".join(
            c for c in filename if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

    def download_author_works(self, author_name: str) -> None:
        """
        Main function to search and download works by an author from Google Scholar.
        """
        print(f"Starting download of works by {author_name} from Google Scholar")

        papers = self.search_author(author_name)

        for paper in papers:
            self.download_paper(paper)

        print(f"\nDownload complete! Found {len(papers)} papers")


# Example usage
if __name__ == "__main__":
    # Create a downloader with English as the default language
    downloader = GoogleScholarDownloader()

    # Or specify a different language, e.g., French
    # downloader = GoogleScholarDownloader(language="fr")

    # Note: Make sure to set the SERPAPI_API_KEY environment variable
    downloader.download_author_works("Albert Einstein")

# Important considerations:
# 1. This implementation uses SerpApi, which is a paid service.
# 2. Make sure to set the SERPAPI_API_KEY environment variable with your API key.
# 3. Be aware of SerpApi's pricing and usage limits.
# 4. This implementation now attempts to download PDF papers.
# 5. Always respect copyright laws and terms of service of any platform you use.
# 6. Be cautious about downloading papers you don't have the right to access.
