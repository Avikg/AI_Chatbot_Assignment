import PySimpleGUI as sg
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np


def extract_text_from_pdf(file_path):
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                for paragraph in text.split("\n\n"):
                    chunks.append({"page": page_number + 1, "text": paragraph})
    return chunks


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def prepare_embeddings(pdf_chunks):
    chunk_texts = [chunk['text'] for chunk in pdf_chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts)
    return chunk_texts, chunk_embeddings

def retrieve_relevant_chunks(question, chunk_texts, chunk_embeddings, pdf_chunks):
    question_embedding = embedding_model.encode(question)
    similarities = cosine_similarity([question_embedding], chunk_embeddings)
    top_indices = np.argsort(similarities[0])[-5:]  # Retrieve top 5 chunks
    return [pdf_chunks[i] for i in top_indices[::-1]]  # Return top chunks in descending order

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def get_answers(question, relevant_chunks):
    answers = []
    for chunk in relevant_chunks:
        try:
            result = qa_model(question=question, context=chunk['text'])
            result['page'] = chunk['page']  # Add page info for reference
            answers.append(result)
        except Exception as e:
            print(f"Error processing chunk on page {chunk['page']}: {e}")
    return sorted(answers, key=lambda x: x['score'], reverse=True)

def main_gui():
    sg.theme("DarkTeal9")

    layout = [
        [sg.Text("PDF Question Answering System", font=("Helvetica", 16))],
        [sg.Text("Select a PDF File:"), sg.Input(key="-PDF-", enable_events=True), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),))],
        [sg.Text("Enter Your Question:"), sg.InputText(key="-QUESTION-")],
        [sg.Button("Load PDF"), sg.Button("Get Answer"), sg.Button("Exit")],
        [sg.Text("Answer:", font=("Helvetica", 12)), sg.Text("", key="-ANSWER-", size=(50, 2), font=("Helvetica", 12), text_color="yellow")],
        [sg.Text("Page:", font=("Helvetica", 12)), sg.Text("", key="-PAGE-", size=(10, 1), font=("Helvetica", 12), text_color="yellow")],
        [sg.Text("Confidence:", font=("Helvetica", 12)), sg.Text("", key="-CONFIDENCE-", size=(10, 1), font=("Helvetica", 12), text_color="yellow")],
    ]

    window = sg.Window("PDF Question Answering System", layout)

    pdf_chunks = None
    chunk_texts = None
    chunk_embeddings = None

    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Load PDF":
            file_path = values["-PDF-"]
            if not file_path:
                sg.popup_error("Please select a PDF file.")
                continue

            try:
                sg.popup("Loading PDF. Please wait...", title="Processing")
                pdf_chunks = extract_text_from_pdf(file_path)
                chunk_texts, chunk_embeddings = prepare_embeddings(pdf_chunks)
                sg.popup("PDF Loaded Successfully!", title="Success")
            except Exception as e:
                sg.popup_error(f"Error loading PDF: {e}")
                pdf_chunks = None

        elif event == "Get Answer":
            if not pdf_chunks or chunk_embeddings is None:  
                sg.popup_error("Please load a PDF file first.")
                continue

            question = values["-QUESTION-"].strip()
            if not question:
                sg.popup_error("Please enter a question.")
                continue

            relevant_chunks = retrieve_relevant_chunks(question, chunk_texts, chunk_embeddings, pdf_chunks)
            answers = get_answers(question, relevant_chunks)

            if answers:
                best_answer = answers[0]
                window["-ANSWER-"].update(best_answer['answer'])
                window["-PAGE-"].update(best_answer['page'])
                window["-CONFIDENCE-"].update(f"{best_answer['score']:.2f}")
            else:
                window["-ANSWER-"].update("No suitable answer found.")
                window["-PAGE-"].update("")
                window["-CONFIDENCE-"].update("")

    window.close()

if __name__ == "__main__":
    main_gui()


    