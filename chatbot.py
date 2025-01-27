import openai
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API-Key einfügen
openai.api_key = "DEIN_API_KEY"

# Funktion: Verarbeitete Daten aus JSON-Datei laden
def load_processed_documents(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Die Datei {file_path} wurde nicht gefunden.")
        return []
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {str(e)}")
        return []

# Funktion: GPT-API aufrufen
def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent für Studien- und Prüfungsordnungen."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Fehler bei der GPT-Anfrage: {str(e)}"

# Funktion: Relevanten Abschnitt im Dokument finden
def find_relevant_section(question, text_chunks):
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(text_chunks + [question])
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        most_similar_index = similarity.argmax()
        return text_chunks[most_similar_index]
    except Exception as e:
        return f"Fehler bei der Abschnittssuche: {str(e)}"

# Interaktiver Chatbot
def interactive_chat():
    text_chunks = load_processed_documents("processed_documents.json")
    if not text_chunks:
        print("Keine verarbeiteten Dokumente gefunden.")
        return
    
    print("Willkommen beim Studien- und Prüfungsordnungs-Chatbot!")
    while True:
        question = input("\nDeine Frage: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Vielen Dank, dass du den Chatbot genutzt hast!")
            break
        answer = ask_gpt(question)
        print(f"\nAntwort: {answer}")

# Hauptprogramm starten
if __name__ == "__main__":
    interactive_chat()
