import logging

logger = logging.getLogger("hallucination_detection")


class CorrectnessLabeler:
    def __init__(self, semantic_model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(semantic_model)
        logger.info(f"Loaded embedder: {semantic_model}")

    def label_substring(self, generated, correct_answers, incorrect_answers):
        """1 if a correct-answer alias appears and no incorrect-answer alias does, 0 otherwise,
        None if the model produced no text."""
        g = generated.lower().strip()
        if not g:
            return None

        matches_correct = any(c.lower().strip() in g for c in correct_answers if c.strip())
        matches_incorrect = any(i.lower().strip() in g for i in incorrect_answers if i.strip())

        return 1 if (matches_correct and not matches_incorrect) else 0

    def label_semantic(self, generated, reference, threshold=0.65):
        """Cosine similarity between generated text and a reference answer via sentence embeddings."""
        from sentence_transformers import util as st_util

        if not generated.strip():
            return 0, 0.0

        emb = self.embedder.encode([generated, reference], convert_to_tensor=True)
        sim = float(st_util.cos_sim(emb[0], emb[1]).item())

        return (1 if sim >= threshold else 0, sim)

    def label_example(self, question_result, dataset_name="triviaqa"):
        """Compute substring + semantic labels for one generated answer."""
        generated = question_result.get("generated", "")
        correct = question_result.get("correct_answers", [])
        incorrect = question_result.get("incorrect_answers", [])
        reference = question_result.get("best_answer") or (correct[0] if correct else "")

        substring = self.label_substring(generated, correct, incorrect)
        semantic, sim = self.label_semantic(generated, reference)

        return {
            "substring_correct": substring,
            "semantic_correct": semantic,
            "semantic_similarity": sim,
        }
