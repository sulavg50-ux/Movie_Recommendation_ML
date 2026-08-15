import pickle
import difflib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, model_path="model/model.pkl"):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.titles = data["titles"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.title_to_index = {t.lower(): i for i, t in enumerate(self.titles)}

    def _resolve_title(self, title: str) -> int | None:
        key = title.lower().strip()
        if key in self.title_to_index:
            return self.title_to_index[key]

        close = difflib.get_close_matches(key, self.title_to_index.keys(), n=1, cutoff=0.6)
        if close:
            return self.title_to_index[close[0]]
        return None

    def similar_to(self, title: str, top_k: int = 10) -> list[str]:
        idx = self._resolve_title(title)
        if idx is None:
            return []

        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        ranked = np.argsort(sim_scores)[::-1]

        results = []
        for i in ranked:
            if i == idx:
                continue
            results.append(self.titles[i])
            if len(results) >= top_k:
                break
        return results

    def recommend_from_history(self, watched_titles: list[str], top_k: int = 10) -> list[str]:
        indices = [self._resolve_title(t) for t in watched_titles]
        indices = [i for i in indices if i is not None]

        if not indices:
            return []

        # average the TF-IDF vectors of everything watched into one "taste profile"
        combined_vector = self.tfidf_matrix[indices].mean(axis=0)
        combined_vector = np.asarray(combined_vector)

        sim_scores = cosine_similarity(combined_vector, self.tfidf_matrix).flatten()
        ranked = np.argsort(sim_scores)[::-1]

        watched_set = set(indices)
        results = []
        for i in ranked:
            if i in watched_set:
                continue
            results.append(self.titles[i])
            if len(results) >= top_k:
                break
        return results