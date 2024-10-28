import feedparser
import requests
import os
from urllib.parse import quote_plus
import json
from time import sleep
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class AuthorWorksDownloader:
    def __init__(self) -> None:
        # Using arXiv and Semantic Scholar APIs as examples of legal, public APIs
        self.arxiv_base: str = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base: str = "https://api.semanticscholar.org/v1"
        self.output_dir: str = "downloaded_works"

    def create_output_dir(self, author_name: str) -> Path:
        """Create directory for downloaded works"""
        author_dir = Path(self.output_dir) / self._sanitize_filename(author_name)
        author_dir.mkdir(parents=True, exist_ok=True)
        return author_dir

    def _sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        return "".join(
            c for c in filename if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()

    def search_arxiv(self, author_name: str) -> List[Dict[str, str]]:
        """Search arXiv for papers by author"""
        print(f"Searching arXiv for papers by {author_name}...")

        query = f'au:"{author_name}"'
        params: Dict[str, Any] = {
            "search_query": query,
            "max_results": 100,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(self.arxiv_base, params=params)
            response.raise_for_status()

            # Parse XML response using feedparser
            feed = feedparser.parse(response.text)
            papers: List[Dict[str, str]] = []

            for entry in feed.entries:
                title = entry.title
                pdf_link = next(
                    (link.href for link in entry.links if link.type == "application/pdf"),
                    None,
                )
                if pdf_link:
                    papers.append({"title": title, "pdf_url": pdf_link})

            return papers

        except requests.exceptions.RequestException as e:
            print(f"Error searching arXiv: {e}")
            return []

    def download_paper(self, paper: Dict[str, str], output_dir: Path) -> None:
        """Download a single paper"""
        try:
            filename = self._sanitize_filename(paper["title"]) + ".pdf"
            output_path = output_dir / filename

            if output_path.exists():
                print(f"File already exists: {filename}")
                return

            print(f"Downloading: {filename}")
            response = requests.get(paper["pdf_url"], stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Be nice to servers
            sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {paper['title']}: {e}")

    def download_author_works(self, author_name: str) -> None:
        """Main function to search and download all works by an author"""
        print(f"Starting download of works by {author_name}")

        # Create output directory
        author_dir = self.create_output_dir(author_name)

        # Search and download from arXiv
        arxiv_papers = self.search_arxiv(author_name)

        # Download papers
        for paper in arxiv_papers:
            self.download_paper(paper, author_dir)

        # Save metadata
        metadata: Dict[str, Any] = {
            "author": author_name,
            "papers_found": len(arxiv_papers),
            "sources_searched": ["arXiv"],
            "download_date": str(datetime.now()),
        }

        with open(author_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDownload complete! Found {len(arxiv_papers)} papers")
        print(f"Files saved in: {author_dir}")


# Example usage
if __name__ == "__main__":
    downloader = AuthorWorksDownloader()

    downloader.download_author_works("William Merrill")
