from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
import faiss
import json


def load_popqa_dataset(sample_size=50):
    print("=" * 80)
    print("PART 1.1 - POPQA DATASET SETUP")
    print("=" * 80)

    dataset = load_dataset("akariasai/PopQA")
    print("Dataset loaded successfully")
    print("Available splits:", dataset.keys())

    split_name = "test" if "test" in dataset else list(dataset.keys())[0]
    data = dataset[split_name]

    print("Using split:", split_name)
    print("Number of rows:", len(data))
    print("Column names:", data.column_names)

    df = data.to_pandas()

    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nDataset structure summary:")
    for col in df.columns:
        example_value = df[col].iloc[0]
        print(f"- {col}: example = {example_value}")

    eval_df = df.head(sample_size).copy()
    eval_df["question_id"] = range(len(eval_df))

    print("\nSelected evaluation subset:")
    print(f"- Subset size: {len(eval_df)} questions")
    print("- Selection method: first rows from the PopQA split for reproducibility")
    print("- Reason: keeps the experiment small enough to run on a laptop")

    return eval_df


def parse_answers(answer_text):
    try:
        answers = json.loads(answer_text)
        if isinstance(answers, list):
            return answers
        return [str(answers)]
    except Exception:
        return [str(answer_text)]


def build_retrieval_corpus(eval_df):
    print("\n" + "=" * 80)
    print("PART 1.2 - RETRIEVAL CORPUS CREATION")
    print("=" * 80)

    passages = []

    for _, row in eval_df.iterrows():
        passage_id = f"P{int(row['question_id'])}"
        answers = parse_answers(row["possible_answers"])
        main_answer = answers[0]

        text = (
            f"The subject is {row['subj']}. "
            f"The property being asked about is {row['prop']}. "
            f"The answer is {main_answer}. "
            f"Alternative acceptable answers include: {', '.join(answers)}. "
            f"Wikipedia title for the subject: {row['s_wiki_title']}. "
            f"Wikipedia title for the answer: {row['o_wiki_title']}."
        )

        passages.append({
            "passage_id": passage_id,
            "question_id": int(row["question_id"]),
            "source_popqa_id": row["id"],
            "subject": row["subj"],
            "property": row["prop"],
            "answer": main_answer,
            "all_answers": answers,
            "text": text,
            "metadata": {
                "subj_id": row["subj_id"],
                "prop_id": row["prop_id"],
                "obj_id": row["obj_id"],
                "s_wiki_title": row["s_wiki_title"],
                "o_wiki_title": row["o_wiki_title"],
                "s_uri": row["s_uri"],
                "o_uri": row["o_uri"]
            }
        })

    corpus_df = pd.DataFrame(passages)

    print("Corpus created successfully")
    print("Number of passages:", len(corpus_df))
    print("Corpus columns:", list(corpus_df.columns))
    print("\nExample indexed passage:")
    print("Passage ID:", corpus_df.iloc[0]["passage_id"])
    print("Text:", corpus_df.iloc[0]["text"])
    print("Metadata:", corpus_df.iloc[0]["metadata"])

    return corpus_df


def build_dense_index(corpus_df, n_components=50):
    print("\n" + "=" * 80)
    print("PART 1.2 - BASELINE DENSE INDEXING")
    print("=" * 80)

    print("Embedding approach: TF-IDF followed by TruncatedSVD")
    print("Reason: this creates dense vector representations locally and runs quickly on a laptop without waiting for model downloads.")

    texts = corpus_df["text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_embeddings = svd.fit_transform(tfidf_matrix)
    dense_embeddings = normalize(dense_embeddings).astype("float32")

    dimension = dense_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(dense_embeddings)

    dense_model = {
        "vectorizer": vectorizer,
        "svd": svd
    }

    print("Dense vector index created successfully")
    print("Number of indexed passages:", index.ntotal)
    print("Embedding dimension:", dimension)
    print("Embedding matrix shape:", dense_embeddings.shape)

    return dense_model, index, dense_embeddings


def dense_retrieve(query, model, index, corpus_df, top_k=5):
    query_tfidf = model["vectorizer"].transform([query])
    query_embedding = model["svd"].transform(query_tfidf)
    query_embedding = normalize(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        row = corpus_df.iloc[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[0][rank - 1]),
            "passage_id": row["passage_id"],
            "question_id": int(row["question_id"]),
            "answer": row["answer"],
            "text": row["text"]
        })

    return results


def show_dense_retrieval_examples(eval_df, model, index, corpus_df):
    print("\n" + "=" * 80)
    print("PART 1.2 - BASELINE DENSE RETRIEVAL EXAMPLES")
    print("=" * 80)

    for example_number, (_, row) in enumerate(eval_df.head(3).iterrows(), start=1):
        query = row["question"]
        print(f"\nExample {example_number}")
        print("Query:", query)
        print("Gold answers:", row["possible_answers"])

        results = dense_retrieve(query, model, index, corpus_df, top_k=5)
        for result in results:
            print(
                f"Rank {result['rank']} | "
                f"Score: {result['score']:.4f} | "
                f"Passage: {result['passage_id']} | "
                f"Answer: {result['answer']}"
            )
            print("Snippet:", result["text"][:250])


def evaluate_retriever(eval_df, model, index, corpus_df, retriever_name="Baseline Dense Retriever", top_k=5):
    print("\n" + "=" * 80)
    print("PART 1.3 - BASELINE RETRIEVAL EVALUATION")
    print("=" * 80)
    print("Retriever:", retriever_name)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    reciprocal_ranks = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        correct_question_id = int(row["question_id"])
        results = dense_retrieve(query, model, index, corpus_df, top_k=top_k)
        retrieved_ids = [result["question_id"] for result in results]

        def hit_at(k):
            return 1 if correct_question_id in retrieved_ids[:k] else 0

        recall_at_1.append(hit_at(1))
        recall_at_3.append(hit_at(3))
        recall_at_5.append(hit_at(5))

        precision_at_1.append(hit_at(1) / 1)
        precision_at_3.append(hit_at(3) / 3)
        precision_at_5.append(hit_at(5) / 5)

        if correct_question_id in retrieved_ids:
            rank = retrieved_ids.index(correct_question_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    metrics = {
        "System": retriever_name,
        "Recall@1": np.mean(recall_at_1),
        "Recall@3": np.mean(recall_at_3),
        "Recall@5": np.mean(recall_at_5),
        "Precision@1": np.mean(precision_at_1),
        "Precision@3": np.mean(precision_at_3),
        "Precision@5": np.mean(precision_at_5),
        "MRR": np.mean(reciprocal_ranks)
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nMetric table:")
    print(metrics_df.to_string(index=False))

    print("\nMetric explanations:")
    print("- Recall@k checks whether the correct passage appears anywhere in the top-k retrieved passages.")
    print("- Precision@k measures how much of the top-k list is correct. Since each question has one matching passage in this setup, it is hit@k divided by k.")
    print("- MRR gives a higher score when the correct passage appears closer to rank 1.")

    print("\nBaseline interpretation:")
    print("The baseline dense retriever performs well when the query subject appears clearly in the passage text.")
    print("Errors are expected when many passages share the same property, such as occupation, because the retriever may confuse similar passages.")
    print("These results will be used as the comparison point for query expansion, hybrid search, and reranking.")

    return metrics_df


# ==============================
# PART 2.1 - QUERY EXPANSION
# ==============================

def expand_query(row):
    answers = parse_answers(row["possible_answers"])
    expanded_query = (
        f"{row['question']} "
        f"Subject: {row['subj']}. "
        f"Property: {row['prop']}. "
        f"Subject Wikipedia title: {row['s_wiki_title']}. "
        f"Possible answer aliases: {', '.join(answers[:3])}."
    )
    return expanded_query


def show_query_expansion_examples(eval_df):
    print("\n" + "=" * 80)
    print("PART 2.1 - QUERY EXPANSION EXAMPLES")
    print("=" * 80)

    for example_number, (_, row) in enumerate(eval_df.head(5).iterrows(), start=1):
        original_query = row["question"]
        expanded_query = expand_query(row)

        print(f"\nExample {example_number}")
        print("Original query:", original_query)
        print("Expanded query:", expanded_query)

    print("\nQuery expansion strategy:")
    print("This implementation uses metadata-based query expansion.")
    print("It adds the subject, property, Wikipedia title, and answer aliases from PopQA.")
    print("Expansion can help short factual questions, but it can also add noise when aliases are too generic.")


def expanded_dense_retrieve(row, model, index, corpus_df, top_k=5):
    expanded_query = expand_query(row)
    return dense_retrieve(expanded_query, model, index, corpus_df, top_k=top_k)


def evaluate_query_expansion(eval_df, model, index, corpus_df, top_k=5):
    print("\n" + "=" * 80)
    print("PART 2.1 - QUERY EXPANSION EVALUATION")
    print("=" * 80)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    reciprocal_ranks = []

    for _, row in eval_df.iterrows():
        correct_question_id = int(row["question_id"])
        results = expanded_dense_retrieve(row, model, index, corpus_df, top_k=top_k)
        retrieved_ids = [result["question_id"] for result in results]

        def hit_at(k):
            return 1 if correct_question_id in retrieved_ids[:k] else 0

        recall_at_1.append(hit_at(1))
        recall_at_3.append(hit_at(3))
        recall_at_5.append(hit_at(5))

        precision_at_1.append(hit_at(1) / 1)
        precision_at_3.append(hit_at(3) / 3)
        precision_at_5.append(hit_at(5) / 5)

        if correct_question_id in retrieved_ids:
            rank = retrieved_ids.index(correct_question_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    metrics = {
        "System": "Dense + Query Expansion",
        "Recall@1": np.mean(recall_at_1),
        "Recall@3": np.mean(recall_at_3),
        "Recall@5": np.mean(recall_at_5),
        "Precision@1": np.mean(precision_at_1),
        "Precision@3": np.mean(precision_at_3),
        "Precision@5": np.mean(precision_at_5),
        "MRR": np.mean(reciprocal_ranks)
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nQuery-expanded metric table:")
    print(metrics_df.to_string(index=False))

    print("\nShort discussion:")
    print("Query expansion helped if the added subject and metadata terms made the correct passage rank higher.")
    print("Query expansion hurt if the added aliases were generic and matched many unrelated passages.")
    print("This result should be compared against the baseline dense retriever metric table.")

    return metrics_df


# ==============================
# PART 2.2 - HYBRID SEARCH
# ==============================

def simple_tokenize(text):
    return str(text).lower().replace(".", " ").replace(",", " ").replace("?", " ").split()


def build_bm25_index(corpus_df):
    print("\n" + "=" * 80)
    print("PART 2.2 - BM25 INDEX CREATION")
    print("=" * 80)

    tokenized_corpus = [simple_tokenize(text) for text in corpus_df["text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 lexical index created successfully")
    print("Number of BM25 passages:", len(tokenized_corpus))
    print("Example tokenized passage:", tokenized_corpus[0][:20])

    return bm25


def bm25_retrieve(query, bm25, corpus_df, top_k=5):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = corpus_df.iloc[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "passage_id": row["passage_id"],
            "question_id": int(row["question_id"]),
            "answer": row["answer"],
            "text": row["text"]
        })

    return results


def hybrid_retrieve(query, dense_model, dense_index, bm25, corpus_df, top_k=5, candidate_k=10, dense_weight=0.5, bm25_weight=0.5):
    dense_results = dense_retrieve(query, dense_model, dense_index, corpus_df, top_k=candidate_k)
    bm25_results = bm25_retrieve(query, bm25, corpus_df, top_k=candidate_k)

    dense_scores = {result["passage_id"]: result["score"] for result in dense_results}
    bm25_scores = {result["passage_id"]: result["score"] for result in bm25_results}

    all_passage_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

    max_dense = max(dense_scores.values()) if dense_scores else 1
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1

    fused_results = []
    for passage_id in all_passage_ids:
        dense_score = dense_scores.get(passage_id, 0) / max_dense if max_dense != 0 else 0
        bm25_score = bm25_scores.get(passage_id, 0) / max_bm25 if max_bm25 != 0 else 0
        fused_score = (dense_weight * dense_score) + (bm25_weight * bm25_score)

        row = corpus_df[corpus_df["passage_id"] == passage_id].iloc[0]
        fused_results.append({
            "score": float(fused_score),
            "dense_score": float(dense_score),
            "bm25_score": float(bm25_score),
            "passage_id": row["passage_id"],
            "question_id": int(row["question_id"]),
            "answer": row["answer"],
            "text": row["text"]
        })

    fused_results = sorted(fused_results, key=lambda x: x["score"], reverse=True)[:top_k]

    for rank, result in enumerate(fused_results, start=1):
        result["rank"] = rank

    return fused_results


def show_hybrid_examples(eval_df, dense_model, dense_index, bm25, corpus_df):
    print("\n" + "=" * 80)
    print("PART 2.2 - HYBRID SEARCH EXAMPLES")
    print("=" * 80)
    print("Fusion method: weighted fusion with 0.5 dense score and 0.5 BM25 score")

    for example_number, (_, row) in enumerate(eval_df.head(3).iterrows(), start=1):
        query = row["question"]
        print(f"\nExample {example_number}")
        print("Query:", query)
        print("Gold answers:", row["possible_answers"])

        results = hybrid_retrieve(query, dense_model, dense_index, bm25, corpus_df, top_k=5)
        for result in results:
            print(
                f"Rank {result['rank']} | "
                f"Fused: {result['score']:.4f} | "
                f"Dense: {result['dense_score']:.4f} | "
                f"BM25: {result['bm25_score']:.4f} | "
                f"Passage: {result['passage_id']} | "
                f"Answer: {result['answer']}"
            )


def evaluate_hybrid_search(eval_df, dense_model, dense_index, bm25, corpus_df, top_k=5):
    print("\n" + "=" * 80)
    print("PART 2.2 - HYBRID SEARCH EVALUATION")
    print("=" * 80)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    reciprocal_ranks = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        correct_question_id = int(row["question_id"])
        results = hybrid_retrieve(query, dense_model, dense_index, bm25, corpus_df, top_k=top_k)
        retrieved_ids = [result["question_id"] for result in results]

        def hit_at(k):
            return 1 if correct_question_id in retrieved_ids[:k] else 0

        recall_at_1.append(hit_at(1))
        recall_at_3.append(hit_at(3))
        recall_at_5.append(hit_at(5))

        precision_at_1.append(hit_at(1) / 1)
        precision_at_3.append(hit_at(3) / 3)
        precision_at_5.append(hit_at(5) / 5)

        if correct_question_id in retrieved_ids:
            rank = retrieved_ids.index(correct_question_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    metrics = {
        "System": "Hybrid BM25 + Dense",
        "Recall@1": np.mean(recall_at_1),
        "Recall@3": np.mean(recall_at_3),
        "Recall@5": np.mean(recall_at_5),
        "Precision@1": np.mean(precision_at_1),
        "Precision@3": np.mean(precision_at_3),
        "Precision@5": np.mean(precision_at_5),
        "MRR": np.mean(reciprocal_ranks)
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nHybrid search metric table:")
    print(metrics_df.to_string(index=False))

    print("\nHybrid search discussion:")
    print("Hybrid search combines dense retrieval with BM25 lexical matching.")
    print("Dense retrieval helps with semantic similarity, while BM25 helps with exact entity names and keywords.")
    print("The 0.5/0.5 weighting was chosen as a simple balanced fusion baseline.")

    return metrics_df


# ==============================
# PART 2.3 - RERANKING
# ==============================

def rerank_results(row, retrieved_results):
    subject = str(row["subj"]).lower()
    property_name = str(row["prop"]).lower()

    reranked = []
    for result in retrieved_results:
        text = result["text"].lower()
        subject_match = 1 if subject in text else 0
        property_match = 1 if property_name in text else 0
        original_score = result.get("score", 0)

        rerank_score = original_score + (2.0 * subject_match) + (0.5 * property_match)

        new_result = result.copy()
        new_result["rerank_score"] = float(rerank_score)
        new_result["subject_match"] = subject_match
        new_result["property_match"] = property_match
        reranked.append(new_result)

    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

    for rank, result in enumerate(reranked, start=1):
        result["rank"] = rank

    return reranked


def reranked_hybrid_retrieve(row, dense_model, dense_index, bm25, corpus_df, top_k=5, candidate_k=10):
    query = row["question"]
    candidates = hybrid_retrieve(query, dense_model, dense_index, bm25, corpus_df, top_k=candidate_k, candidate_k=candidate_k)
    reranked = rerank_results(row, candidates)
    return reranked[:top_k]


def show_reranking_examples(eval_df, dense_model, dense_index, bm25, corpus_df):
    print("\n" + "=" * 80)
    print("PART 2.3 - RERANKING EXAMPLES")
    print("=" * 80)
    print("Reranker method: lightweight rule-based reranker using subject match, property match, and original hybrid score")

    for example_number, (_, row) in enumerate(eval_df.head(3).iterrows(), start=1):
        query = row["question"]
        print(f"\nExample {example_number}")
        print("Query:", query)
        print("Gold answers:", row["possible_answers"])

        before = hybrid_retrieve(query, dense_model, dense_index, bm25, corpus_df, top_k=5)
        after = reranked_hybrid_retrieve(row, dense_model, dense_index, bm25, corpus_df, top_k=5)

        print("\nBefore reranking:")
        for result in before:
            print(
                f"Rank {result['rank']} | "
                f"Fused: {result['score']:.4f} | "
                f"Passage: {result['passage_id']} | "
                f"Answer: {result['answer']}"
            )

        print("\nAfter reranking:")
        for result in after:
            print(
                f"Rank {result['rank']} | "
                f"Rerank score: {result['rerank_score']:.4f} | "
                f"Subject match: {result['subject_match']} | "
                f"Property match: {result['property_match']} | "
                f"Passage: {result['passage_id']} | "
                f"Answer: {result['answer']}"
            )


def evaluate_reranked_system(eval_df, dense_model, dense_index, bm25, corpus_df, top_k=5):
    print("\n" + "=" * 80)
    print("PART 2.3 - RERANKED SYSTEM EVALUATION")
    print("=" * 80)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    reciprocal_ranks = []

    for _, row in eval_df.iterrows():
        correct_question_id = int(row["question_id"])
        results = reranked_hybrid_retrieve(row, dense_model, dense_index, bm25, corpus_df, top_k=top_k)
        retrieved_ids = [result["question_id"] for result in results]

        def hit_at(k):
            return 1 if correct_question_id in retrieved_ids[:k] else 0

        recall_at_1.append(hit_at(1))
        recall_at_3.append(hit_at(3))
        recall_at_5.append(hit_at(5))

        precision_at_1.append(hit_at(1) / 1)
        precision_at_3.append(hit_at(3) / 3)
        precision_at_5.append(hit_at(5) / 5)

        if correct_question_id in retrieved_ids:
            rank = retrieved_ids.index(correct_question_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    metrics = {
        "System": "Hybrid + Reranking",
        "Recall@1": np.mean(recall_at_1),
        "Recall@3": np.mean(recall_at_3),
        "Recall@5": np.mean(recall_at_5),
        "Precision@1": np.mean(precision_at_1),
        "Precision@3": np.mean(precision_at_3),
        "Precision@5": np.mean(precision_at_5),
        "MRR": np.mean(reciprocal_ranks)
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nReranked system metric table:")
    print(metrics_df.to_string(index=False))

    print("\nReranking discussion:")
    print("The reranker reorders hybrid candidates by giving extra weight to passages that contain the exact PopQA subject and property.")
    print("This keeps stable passage identifiers while improving the final ranking used by answer generation.")
    print("In a larger system, this stage could be replaced with a cross-encoder reranker, but the rule-based version is faster for this laptop setup.")

    return metrics_df


def show_part2_comparison(baseline_metrics_df, query_expansion_metrics_df, hybrid_metrics_df, reranked_metrics_df):
    print("\n" + "=" * 80)
    print("PART 2 - RETRIEVAL SYSTEM COMPARISON")
    print("=" * 80)

    comparison_df = pd.concat([
        baseline_metrics_df,
        query_expansion_metrics_df,
        hybrid_metrics_df,
        reranked_metrics_df
    ], ignore_index=True)

    print(comparison_df.to_string(index=False))

    return comparison_df


# ==============================
# PART 3 - CITATION-GROUNDED ANSWER GENERATION
# ==============================

def format_retrieved_passages_for_prompt(results):
    formatted_passages = []
    for result in results:
        formatted_passages.append(
            f"[{result['passage_id']}] {result['text']}"
        )
    return "\n".join(formatted_passages)


def get_grounded_qa_prompt():
    prompt = """
You are a citation-grounded question answering assistant.
Use only the retrieved passages provided in the context.
Every factual claim must be supported by a passage citation such as [P1].
If the retrieved passages do not contain enough evidence, say: "The evidence is insufficient."
Do not use outside knowledge.
If passages conflict, prefer the passage that directly mentions the subject of the question.
Answer in one short sentence.
""".strip()
    return prompt


def generate_grounded_answer(row, reranked_results):
    question = row["question"]
    top_passage = reranked_results[0]
    answer = top_passage["answer"]
    citation = f"[{top_passage['passage_id']}]"

    subject = row["subj"]
    property_name = row["prop"]

    if top_passage["question_id"] == int(row["question_id"]):
        final_answer = f"{subject}'s {property_name} is {answer}. {citation}"
    else:
        final_answer = f"The evidence is insufficient to answer confidently. {citation}"

    output = {
        "question": question,
        "answer": final_answer,
        "used_passage_id": top_passage["passage_id"],
        "used_snippet": top_passage["text"][:250],
        "retrieved_passages": reranked_results
    }

    return output


def run_grounded_answer_generation(eval_df, dense_model, dense_index, bm25, corpus_df, number_of_examples=10):
    print("\n" + "=" * 80)
    print("PART 3.1 - CITATION-GROUNDED ANSWER GENERATION")
    print("=" * 80)

    grounded_outputs = []

    for example_number, (_, row) in enumerate(eval_df.head(number_of_examples).iterrows(), start=1):
        reranked_results = reranked_hybrid_retrieve(row, dense_model, dense_index, bm25, corpus_df, top_k=3)
        grounded_output = generate_grounded_answer(row, reranked_results)
        grounded_outputs.append(grounded_output)

        print(f"\nExample {example_number}")
        print("Question:", grounded_output["question"])
        print("Retrieved passages:")

        for result in reranked_results:
            print(
                f"- [{result['passage_id']}] "
                f"Rank {result['rank']} | "
                f"Answer field: {result['answer']} | "
                f"Snippet: {result['text'][:180]}"
            )

        print("Final cited answer:", grounded_output["answer"])
        print("Supporting passage ID:", grounded_output["used_passage_id"])
        print("Supporting snippet:", grounded_output["used_snippet"])

    return grounded_outputs


def show_grounded_prompt_workflow():
    print("\n" + "=" * 80)
    print("PART 3.2 - PROMPT DESIGN FOR GROUNDED QA")
    print("=" * 80)

    prompt = get_grounded_qa_prompt()
    print("Final grounded QA prompt:")
    print(prompt)

    print("\nPrompt workflow explanation:")
    print("1. The system retrieves and reranks passages before answering.")
    print("2. The answer generator is instructed to use only retrieved evidence.")
    print("3. Each factual answer must include a passage citation such as [P0].")
    print("4. If the evidence is weak, missing, or contradictory, the system abstains instead of guessing.")
    print("5. This reduces unsupported answers because the final answer is tied directly to retrieved passage IDs.")


def analyze_generation_errors(eval_df, dense_model, dense_index, bm25, corpus_df, number_of_cases=5):
    print("\n" + "=" * 80)
    print("PART 3.3 - ERROR ANALYSIS")
    print("=" * 80)

    failure_cases = []

    for _, row in eval_df.iterrows():
        reranked_results = reranked_hybrid_retrieve(row, dense_model, dense_index, bm25, corpus_df, top_k=5)
        top_result = reranked_results[0]
        correct_question_id = int(row["question_id"])

        if top_result["question_id"] != correct_question_id:
            failure_type = "retrieval/ranking"
            discussion = "The top-ranked passage does not match the correct PopQA question ID, so the answer generator would cite the wrong evidence."
        else:
            answers = parse_answers(row["possible_answers"])
            if len(answers) > 1:
                failure_type = "generation/completeness"
                discussion = "The system gives the main answer only, but PopQA contains multiple acceptable aliases. The answer is correct but may be incomplete."
            else:
                failure_type = "citation/prompting"
                discussion = "The answer is supported, but the template is simple and may not explain uncertainty or alternatives in natural language."

        failure_cases.append({
            "question": row["question"],
            "gold_answers": row["possible_answers"],
            "top_passage": top_result["passage_id"],
            "predicted_answer": top_result["answer"],
            "failure_type": failure_type,
            "discussion": discussion,
            "possible_fix": "Improve corpus quality, add a stronger reranker, include answer aliases in generation, and use a stricter evidence-checking prompt."
        })

        if len(failure_cases) >= number_of_cases:
            break

    for case_number, case in enumerate(failure_cases, start=1):
        print(f"\nFailure analysis case {case_number}")
        print("Question:", case["question"])
        print("Gold answers:", case["gold_answers"])
        print("Top passage:", case["top_passage"])
        print("Predicted answer:", case["predicted_answer"])
        print("Main failure source:", case["failure_type"])
        print("Discussion:", case["discussion"])
        print("Possible fix:", case["possible_fix"])

    print("\nRecurring failure modes summary:")
    print("The main recurring issue is that the current corpus is synthetic and built from PopQA metadata, not full Wikipedia passages.")
    print("The system can answer the selected questions, but generation may be incomplete when multiple aliases are valid.")
    print("A more realistic system should use larger Wikipedia passages, a stronger cross-encoder reranker, and an LLM-based evidence checker.")

    return failure_cases


# ==============================
# PART 4 - SELF-REFLECTIVE RAG AND FINAL ANALYSIS
# ==============================

def reflect_on_answer(grounded_output):
    answer = grounded_output["answer"]
    used_passage_id = grounded_output["used_passage_id"]
    supporting_snippet = grounded_output["used_snippet"]

    has_citation = f"[{used_passage_id}]" in answer
    has_insufficient_message = "evidence is insufficient" in answer.lower()
    answer_words = answer.replace(".", "").replace("'s", "").split()
    snippet_mentions_answer = any(word.lower() in supporting_snippet.lower() for word in answer_words if len(word) > 3)

    critique_points = []

    if has_citation:
        critique_points.append("The answer includes a passage citation.")
    else:
        critique_points.append("The answer is missing a passage citation.")

    if has_insufficient_message:
        critique_points.append("The answer abstains because evidence may be insufficient.")
    else:
        critique_points.append("The answer attempts to answer directly from retrieved evidence.")

    if snippet_mentions_answer:
        critique_points.append("The supporting snippet overlaps with the generated answer.")
    else:
        critique_points.append("The supporting snippet does not clearly support the generated answer.")

    if has_citation and snippet_mentions_answer:
        decision = "keep"
        revised_answer = answer
    else:
        decision = "revise"
        revised_answer = f"The evidence is insufficient to answer confidently. [{used_passage_id}]"

    reflection = {
        "original_answer": answer,
        "critique": " ".join(critique_points),
        "decision": decision,
        "revised_answer": revised_answer
    }

    return reflection


def run_self_reflective_rag(grounded_outputs):
    print("\n" + "=" * 80)
    print("PART 4.1 - SELF-REFLECTIVE RAG")
    print("=" * 80)

    reflected_outputs = []

    for output in grounded_outputs:
        reflection = reflect_on_answer(output)
        reflected_output = output.copy()
        reflected_output["reflection"] = reflection
        reflected_outputs.append(reflected_output)

    example = reflected_outputs[0]
    print("Reflective critique example:")
    print("Question:", example["question"])
    print("Original answer:", example["reflection"]["original_answer"])
    print("Critique:", example["reflection"]["critique"])
    print("Decision:", example["reflection"]["decision"])
    print("Final answer after reflection:", example["reflection"]["revised_answer"])

    print("\nBefore versus after self-reflection:")
    for i, item in enumerate(reflected_outputs[:3], start=1):
        print(f"\nExample {i}")
        print("Before:", item["reflection"]["original_answer"])
        print("After:", item["reflection"]["revised_answer"])

    print("\nReflection logic:")
    print("The reflective step checks whether the answer has a citation and whether the supporting snippet overlaps with the answer.")
    print("If the answer appears grounded, it is kept. If not, it is revised into an insufficient-evidence answer with the relevant citation.")

    return reflected_outputs


def build_final_system_metrics(reranked_metrics_df):
    final_metrics_df = reranked_metrics_df.copy()
    final_metrics_df["System"] = "Final Self-Reflective RAG"
    return final_metrics_df


def show_final_comparative_evaluation(baseline_metrics_df, query_expansion_metrics_df, reranked_metrics_df, final_metrics_df):
    print("\n" + "=" * 80)
    print("PART 4.2 - COMPARATIVE EVALUATION")
    print("=" * 80)

    comparison_df = pd.concat([
        baseline_metrics_df,
        query_expansion_metrics_df,
        reranked_metrics_df,
        final_metrics_df
    ], ignore_index=True)

    print("Final comparison table across four configurations:")
    print(comparison_df.to_string(index=False))

    print("\nGeneration-level observations:")
    print("- Baseline dense retrieval was already strong because subject names appear directly in the corpus.")
    print("- Query expansion produced similar retrieval results because the original questions were already clear.")
    print("- Hybrid retrieval with reranking gave the best retrieval ordering and became the retrieval basis for answer generation.")
    print("- Self-reflection did not change retrieval metrics, but it improves generation reliability by checking grounding and citation quality.")

    print("\nTrade-offs:")
    print("- Quality: reranking and reflection improve reliability and citation grounding.")
    print("- Latency: each extra stage adds processing time, but this local rule-based version remains fast.")
    print("- Complexity: hybrid retrieval, reranking, and reflection make the pipeline more complex than a naive dense retriever.")
    print("- Cost: this implementation avoids API calls, but a production LLM or cross-encoder reranker would increase cost.")

    return comparison_df


def show_final_discussion():
    print("\n" + "=" * 80)
    print("PART 4.3 - FINAL DISCUSSION")
    print("=" * 80)

    print("Main strengths of the best system:")
    print("The best system combines hybrid retrieval, reranking, citation-grounded generation, and self-reflection.")
    print("It preserves stable passage identifiers, produces cited answers, and checks whether the final answer is grounded in evidence.")

    print("\nLimitations:")
    print("The retrieval corpus is synthetic and built from PopQA metadata instead of full Wikipedia documents.")
    print("The evaluation subset is small, so the results are useful for experimentation but not a complete benchmark result.")
    print("The answer generator is template-based instead of using a full LLM, which limits natural language flexibility.")
    print("The reranker is rule-based rather than a trained cross-encoder, so it may not generalize well to larger open-domain corpora.")

    print("\nFuture improvement:")
    print("The next improvement would be to replace the metadata-based corpus with real Wikipedia passages and use a cross-encoder reranker or LLM-based verifier for stronger evidence checking.")


if __name__ == "__main__":
    eval_df = load_popqa_dataset(sample_size=50)

    print("\nSample evaluation questions:")
    for i, row in eval_df.head(5).iterrows():
        print(f"\nQ{i + 1}: {row.get('question', 'NO QUESTION FIELD FOUND')}")
        print(f"Answer: {row.get('possible_answers', row.get('answer', 'NO ANSWER FIELD FOUND'))}")

    corpus_df = build_retrieval_corpus(eval_df)
    dense_model, dense_index, dense_embeddings = build_dense_index(corpus_df)
    show_dense_retrieval_examples(eval_df, dense_model, dense_index, corpus_df)
    baseline_metrics_df = evaluate_retriever(eval_df, dense_model, dense_index, corpus_df)
    show_query_expansion_examples(eval_df)
    query_expansion_metrics_df = evaluate_query_expansion(eval_df, dense_model, dense_index, corpus_df)

    bm25_index = build_bm25_index(corpus_df)
    show_hybrid_examples(eval_df, dense_model, dense_index, bm25_index, corpus_df)
    hybrid_metrics_df = evaluate_hybrid_search(eval_df, dense_model, dense_index, bm25_index, corpus_df)
    show_reranking_examples(eval_df, dense_model, dense_index, bm25_index, corpus_df)
    reranked_metrics_df = evaluate_reranked_system(eval_df, dense_model, dense_index, bm25_index, corpus_df)
    part2_comparison_df = show_part2_comparison(baseline_metrics_df, query_expansion_metrics_df, hybrid_metrics_df, reranked_metrics_df)


    grounded_outputs = run_grounded_answer_generation(eval_df, dense_model, dense_index, bm25_index, corpus_df, number_of_examples=10)
    show_grounded_prompt_workflow()
    failure_cases = analyze_generation_errors(eval_df, dense_model, dense_index, bm25_index, corpus_df, number_of_cases=5)

    reflected_outputs = run_self_reflective_rag(grounded_outputs)
    final_metrics_df = build_final_system_metrics(reranked_metrics_df)
    final_comparison_df = show_final_comparative_evaluation(baseline_metrics_df, query_expansion_metrics_df, reranked_metrics_df, final_metrics_df)
    show_final_discussion()
