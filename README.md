# PDF Question Answering System

## Overview
The PDF Question Answering System extracts text from PDF files, processes the content into smaller chunks, creates embeddings for efficient text similarity, and uses pre-trained models to answer user queries. A user-friendly graphical interface (GUI) built with `PySimpleGUI` simplifies interaction.

## Features

1. **PDF File Loading**:
   - Users can select and load a PDF file.
   - Text is extracted and split into manageable chunks for processing.
   - Displays a success message after loading the file.

2. **Question Answering**:
   - Users input a question, and the system retrieves relevant text chunks.
   - A pre-trained `transformers` QA model processes the chunks to generate answers.

3. **Results Display**:
   - Best answer, page number, and confidence score are displayed in the GUI.

4. **Error Handling**:
   - Handles missing files, empty questions, and processing errors gracefully with user-friendly messages.

---

## System Components

### 1. PDF Text Extraction
- **Library**: `pdfplumber`
- **Functionality**:
  - Extracts text from each page of the PDF.
  - Splits text into smaller chunks by paragraphs (`\n\n`) for efficient processing.

### 2. Embedding Model
- **Library**: `sentence-transformers`
- **Model**: `all-MiniLM-L6-v2`
- **Functionality**:
  - Encodes extracted text chunks into numerical vectors (embeddings).
  - Encodes user questions into embeddings to compute similarity.

### 3. Question-Answering Model
- **Library**: `transformers`
- **Model**: `distilbert-base-cased-distilled-squad`
- **Functionality**:
  - Extracts the most relevant text snippet for answering a question.
  - Scores and ranks answers by confidence.

### 4. Similarity Computation
- **Library**: `sklearn`
- **Metric**: Cosine Similarity
- **Functionality**:
  - Compares question embeddings with chunk embeddings.
  - Selects the top 5 most similar chunks.

### 5. GUI
- **Library**: `PySimpleGUI`
- **Functionality**:
  - Allows users to:
    - Select a PDF file.
    - Input a question.
    - View answers, page numbers, and confidence scores.
  - Handles errors like missing files or questions.

---

## Code Structure

| Function                     | Purpose                                                            |
|------------------------------|--------------------------------------------------------------------|
| `extract_text_from_pdf`      | Extracts text from all pages of a given PDF file.                  |
| `prepare_embeddings`         | Creates embeddings for text chunks using the `sentence-transformers` model. |
| `retrieve_relevant_chunks`   | Finds the most relevant chunks for a user question using cosine similarity. |
| `get_answers`                | Uses the QA model to generate answers from the most relevant chunks. |
| `main_gui`                   | Implements the GUI logic for file selection, question answering, and results display. |

---

## Step-by-Step Workflow

1. **Load a PDF File**:
   - The user selects a file using the file selector in the GUI.
   - The `Load PDF` button triggers:
     - `extract_text_from_pdf`: Extracts and processes text from the PDF.
     - `prepare_embeddings`: Creates embeddings for text chunks.

2. **Ask a Question**:
   - The user enters a question in the input field.
   - The `Get Answer` button triggers:
     - `retrieve_relevant_chunks`: Retrieves the top 5 most relevant chunks based on cosine similarity.
     - `get_answers`: Uses the QA model to generate and rank answers.

3. **Display Results**:
   - The best answer is displayed in the GUI along with:
     - Page number of the chunk containing the answer.
     - Confidence score of the model’s prediction.

4. **Handle Errors**:
   - Missing file:
     - Prompts the user to select a file before processing.
   - Empty question:
     - Displays an error message asking for a valid question.
   - Processing errors:
     - Catches exceptions and displays error messages in the GUI.


## Usage Instructions

**Run the Application**:
   ```bash
   python3 model.py
