import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    return True

load_nltk()

MAX_SENTENCES_PER_CHUNK = 5
OVERLAP = 1

@st.cache_resource
def load_data():
    df = pd.read_csv("papers.csv")
    df["clean_text"] = df["Текст статьи"].apply(lambda x: re.sub(r"\s+", " ", x.strip()))
    df_temp = pd.read_csv("partial_results.csv")
    df_temp['question_clean'] = df_temp['question']

    new_rows = df_temp[['title', 'chunk']].drop_duplicates().copy()
    new_rows['question'] = new_rows['title']
    new_rows['question_clean'] = new_rows['title']
    final_df = pd.concat([df_temp, new_rows], ignore_index=True)

    questions = final_df["question_clean"].tolist()
    return final_df, questions

final_df, questions = load_data()

@st.cache_resource
def load_models_and_indexes(questions):
    embeddings = np.load('embeddings_titles.npy')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

    tokenized_questions = [q.split() for q in questions]
    bm25 = BM25Okapi(tokenized_questions)

    qa_tokenizer = AutoTokenizer.from_pretrained("KirrAno93/rubert-base-cased-finetuned-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("KirrAno93/rubert-base-cased-finetuned-squad")
    qa_pipeline_model = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

    model = SentenceTransformer('sentence-transformers/LaBSE')

    return model, embeddings, index, tfidf_vectorizer, tfidf_matrix, bm25, qa_pipeline_model

model, embeddings, index, tfidf_vectorizer, tfidf_matrix, bm25, qa_pipeline = load_models_and_indexes(questions=questions)

def retrieve_top_k_embeddings(query, top_k=5):
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, top_k)
    return [(int(idx_), float(distances[0][i])) for i, idx_ in enumerate(indices[0])]

def compute_embedding_score(query, doc_idx):
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    doc_vec = embeddings[doc_idx].reshape(1, -1)
    score = float((q_vec @ doc_vec.T)[0][0])
    return score

def retrieve_top_k_tfidf(query, top_k=5):
    query_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [(int(idx), float(sims[idx])) for idx in top_indices]

def compute_tfidf_score(query, doc_idx):
    query_vec = tfidf_vectorizer.transform([query])
    doc_vec = tfidf_matrix[doc_idx]
    sims = cosine_similarity(query_vec, doc_vec)[0][0]
    return float(sims)

def retrieve_top_k_bm25(query, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

def compute_bm25_score(query, doc_idx):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    return float(scores[doc_idx])

def retrieve_ensemble(query, top_k=5):
    emb_results = retrieve_top_k_embeddings(query, top_k=top_k)
    tfidf_results = retrieve_top_k_tfidf(query, top_k=top_k)
    bm25_results = retrieve_top_k_bm25(query, top_k=top_k)

    candidate_indices = set([x[0] for x in emb_results] + [x[0] for x in tfidf_results] + [x[0] for x in bm25_results])

    final_results = []
    for idx_ in candidate_indices:
        emb_score = compute_embedding_score(query, idx_)
        tfidf_score = compute_tfidf_score(query, idx_)
        bm25_score = compute_bm25_score(query, idx_)

        final_score = (emb_score + tfidf_score + bm25_score) / 3.0
        final_results.append((idx_, final_score, emb_score, tfidf_score, bm25_score))

    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results

def get_all_chunks_for_user_question(user_query, top_k=5):
    ensemble_results = retrieve_ensemble(user_query, top_k=top_k)
    candidates = []

    for r in ensemble_results:
        doc_idx, final_score, emb_s, tfidf_s, bm25_s = r
        title = final_df.iloc[doc_idx]["title"]
        chunk = final_df.iloc[doc_idx]["chunk"]
        question = final_df.iloc[doc_idx]["question"]

        candidates.append({
            "doc_idx": doc_idx,
            "final_score": final_score,
            "emb_score": emb_s,
            "tfidf_score": tfidf_s,
            "bm25_score": bm25_s,
            "title": title,
            "chunk": chunk,
            "question": question
        })
    sorted_candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    return sorted_candidates[:top_k]

def generate_answer(question, top_docs):
    context = " ".join([doc["chunk"] for doc in top_docs])
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

st.title("Чат по статьям")

user_question = st.text_input("Введите ваш вопрос", "")
if st.button("Отправить"):
    with st.spinner("Ищу ответы..."):
        candidates = get_all_chunks_for_user_question(user_question, top_k=5)
        answer = generate_answer(user_question, candidates)

    st.write("**Ваш вопрос:**", user_question)
    st.write("**Ответ:**", answer)

    st.write("**Топ-5 кандидатов:**")
    for c in candidates:
        st.write(f"**Score:** {c['final_score']:.4f}\n**Title:** {c['title']}\n**Question:** {c['question']}")
        st.write(f"Chunk: {c['chunk']}\n")
