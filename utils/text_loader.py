def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
