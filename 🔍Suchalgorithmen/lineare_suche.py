from database import data

search = "KI"

for rows in data:
    split_words = rows["text"].lower().split(" ") # Liste der einzelnden Wörter in der Zeile
    if search.lower() in split_words:
        print(f'Das Wort: "{search}" wurde gefunden in: {rows["id"]}')