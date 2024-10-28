import os
from typing import List, Optional
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from pydantic import BaseModel, Field
from typing import List


class ZettelkastenNote(BaseModel):
    """https://zettelkasten.de/posts/create-zettel-from-reading-notes/"""

    title: str = Field(
        description="A brief, descriptive title summarizing the main focus or topic of the note.",
        example="Understanding the Zettelkasten Method",
    )
    content: str = Field(
        description="Main content of the note, containing core information or insights."
        "The content should be *scoped*, *atomic* and *composable*."
        " - *Scoped*: The content should be about ONE SINGLE idea, not multiple ideas."
        " - *Atomic*: you should be able to read a note in isolation and understand the main point without needing any other notes or the original source material."
        " - *Composable*: you should be able to compose this note with other notes to form a coherent and comprehensive understanding of the topic. You should also be able to compose this note with other notes to form a new, more complex idea.",
        example="The Zettelkasten method is a knowledge management system designed to help retain and organize knowledge. It is a system of interconnected notes that form a network of ideas. It enables you to explore connections between ideas and to build a deep understanding of a topic by connecting related ideas systematically.",
    )
    tags: List[str] = Field(
        description="Tags or keywords used to categorize the note and help with search and organization.",
        example=["Zettelkasten"],
    )


class ZettelkastenNoteWithMetadata(ZettelkastenNote):
    note_id: str = Field(
        description="A unique identifier for the note.",
        exclude=True,
    )
    created_at: datetime = Field(
        description="The date and time when the note was created.",
        exclude=True,
    )


class CreateZettelkastenNoteTool(BaseModel):
    ZettelkastenNotes: List[ZettelkastenNote] = Field(
        description="Represents a list of notes within a Zettelkasten system. "
        "Each note functions as a knowledge unit, connecting ideas and insights "
        "systematically for effective knowledge retention and discovery.",
    )


def extract_key_points(
    markdown_content: str,
) -> Optional[list[ZettelkastenNoteWithMetadata]]:
    """Extract key points from markdown content using OpenAI."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        response_format=CreateZettelkastenNoteTool,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts key points from text and creates concise summaries.",
            },
            {
                "role": "user",
                "content": f"Extract the main ideas and key points from the following markdown content and format them as Zettelkasten notes:\n\n{markdown_content}",
            },
        ],
    )
    if (
        response
        and response.choices
        and response.choices[0]
        and response.choices[0].message
        and response.choices[0].message.parsed
    ):
        notes = response.choices[0].message.parsed

        notes_with_metadata = []
        for note in notes.ZettelkastenNotes:
            note_with_metadata = ZettelkastenNoteWithMetadata(
                **note.model_dump(),
                note_id=str(uuid.uuid4())[:8],
                created_at=datetime.now(),
            )
            notes_with_metadata.append(note_with_metadata)

        return notes_with_metadata
    else:
        return None


def format_zettelkasten_note(note: ZettelkastenNoteWithMetadata) -> str:
    """Format a ZettelkastenNote into a nicely formatted markdown note."""

    formatted_note = f"""# {note.note_id} {note.title}

## Content
{note.content}

## Metadata
- **Created**: {note.created_at}
- **Tags**: {', '.join(f'#{tag}' for tag in note.tags)}

## Links
<!-- Add links to related notes here -->

## References
<!-- Add references or sources here -->

"""
    return formatted_note


def extract_overall_summary(document: str) -> Optional[str]:
    """Extract an overall summary of the document."""
    # make openai call to extract overall summary
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts an overall summary of the document.",
            },
            {
                "role": "user",
                "content": f"Extract an overall summary of the following document:\n\n{document}",
            },
        ],
    )

    return response.choices[0].message.content


def create_head_note(
    *, document_name: str, document: str, notes: List[ZettelkastenNoteWithMetadata]
) -> str:
    """Create a head note that links to all other notes."""

    # add an overall summary of the document.
    overall_summary = extract_overall_summary(document)

    # add links for all the notes
    links = "\n".join(f"- [[{note.title.replace(' ', '_')}]]" for note in notes)

    return f"# {document_name}\n\n{overall_summary}\n\n{links}"


if __name__ == "__main__":
    markdown_file: str = "output_markdown/A Formal Hierarchy of RNN Architectures.md"
    output_dir: str = "obsidian_notes"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(markdown_file, "r", encoding="utf-8") as f:
        markdown_content: str = f.read()

    key_points: Optional[list[ZettelkastenNoteWithMetadata]] = extract_key_points(
        markdown_content
    )

    if key_points is not None:
        for note in key_points:
            formatted_note = format_zettelkasten_note(note)
            filename = f"{note.title.replace(' ', '_')}.md"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(formatted_note)
        print(f"Created {len(key_points)} Zettelkasten notes in {output_dir}")

        head_note = create_head_note(
            document_name=os.path.basename(markdown_file),
            document=markdown_content,
            notes=key_points,
        )
        with open(
            os.path.join(output_dir, f"HEAD_NOTE_{os.path.basename(markdown_file)}"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(head_note)
    else:
        print("Failed to extract key points from the markdown content.")
