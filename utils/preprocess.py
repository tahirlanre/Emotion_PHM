import re


def preprocess_text(text):
    text = re.sub(r"@\S+", " <user> ", text)  # remove user mentions
    text = re.sub(r"http\S+", " <url> ", text)  # remove
    text = re.sub(r"\s+", " ", text)  # remove double or more space
    text = text.strip()
    return text


def preprocess(dataf):
    dataf = dataf.copy()

    dataf["text"] = dataf.text.apply(preprocess_text)
    return dataf
