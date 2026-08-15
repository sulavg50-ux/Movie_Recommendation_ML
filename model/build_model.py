import ast
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def parse_genres(raw: str) -> str:
    try:
        items = ast.literal_eval(raw)
        return " ".join(i["name"] for i in items)
    except (ValueError, SyntaxError):
        return ""


def build():
    df = pd.read_csv("data/movies_fixed.csv", low_memory=False)
    df = df[["title", "overview", "genres", "tagline", "vote_average", "popularity"]]
    df = df.dropna(subset=["title"])
    df = df.reset_index(drop=True)

    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres"] = df["genres"].apply(parse_genres)

    df["tags"] = df["overview"] + " " + df["genres"] + " " + df["tagline"]
    df["tags"] = df["tags"].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    with open("model/model.pkl", "wb") as f:
        pickle.dump(
            {
                "titles": df["title"].tolist(),
                "tfidf": tfidf,
                "tfidf_matrix": tfidf_matrix,
            },
            f,
        )

    print(f"Built model with {len(df)} movies, vocabulary size {len(tfidf.vocabulary_)}")


if __name__ == "__main__":
    build()