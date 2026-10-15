import re

def preprocess_texts(texts):
    cleaned_texts = []

    for text in texts:
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        cleaned_texts.append(text)

    return cleaned_texts