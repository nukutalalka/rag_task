import pandas as pd
import re
import nltk
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import signal
import logging

nltk.download('punkt')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)

# Параметры чанков
MAX_SENTENCES_PER_CHUNK = 5
OVERLAP = 1

REQUESTS_PER_MINUTE = 500
SECONDS_PER_MINUTE = 60
PAUSE_BETWEEN_BATCHES = SECONDS_PER_MINUTE / REQUESTS_PER_MINUTE  # Пауза между запросами

results = []
partial_save_file = "partial_results.csv"

# Чтение файла
df = pd.read_csv("papers.csv")

# Чистка текста
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

df["clean_text"] = df["Текст статьи"].apply(clean_text)

# Разбиение на чанки по предложениям
def chunk_text_by_sentences(text, max_sentences=5, overlap=1):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunk_sents = sentences[start:end]
        chunk_text = " ".join(chunk_sents)
        chunks.append(chunk_text)
        new_start = end - overlap
        if new_start <= start:
            break
        start = new_start
    return chunks

# Создание списка чанков
docs = []
for idx, row in df.iterrows():
    title = row["Заголовок"]
    text = row["clean_text"]
    text_chunks = chunk_text_by_sentences(text, max_sentences=MAX_SENTENCES_PER_CHUNK, overlap=OVERLAP)
    for ch in text_chunks:
        docs.append({
            "title": title,
            "text_chunk": ch
        })

#texts_for_embedding = [(d["title"] + " " + d["text_chunk"]) for d in docs]
texts_for_embedding = [(d["title"], d["text_chunk"]) for d in docs]


# Функция для генерации вопросов через ChatGPT API
def generate_questions_with_chatgpt(title, chunk, api_key):
    client = OpenAI(api_key=api_key)
    prompt = f"Сгенерируй ровно 5 вопросов по следующему контексту:\n\n{chunk}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты помощник, который генерирует ровно 5 вопросов по предоставленному контексту."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        response_text = response.choices[0].message.content
        questions = [q.strip() for q in response_text.split("\n") if re.match(r'^\d+\.', q)]
        return [{"title": title, "chunk": chunk, "question": q} for q in questions[:5]]
    except Exception as e:
        print(f"Ошибка при обработке чанка: {e}")
        return []

# Генерация вопросов
API_KEY = ""
results = []

def process_chunks_with_rate_limit(chunks, api_key, max_workers=10):
    global results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (title, chunk) in enumerate(chunks):
            logging.info(f"Отправка запроса {i + 1}/{len(chunks)}")
            futures.append(executor.submit(generate_questions_with_chatgpt, title, chunk, api_key))
            time.sleep(PAUSE_BETWEEN_BATCHES)

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.extend(result)
                    # Автосохранение на каждом этапе
                    save_partial_results()
                    logging.info(f"Успешно обработан чанк. Всего сохранено результатов: {len(results)}")
            except Exception as e:
                logging.error(f"Ошибка в процессе: {e}")

def save_partial_results():
    pd.DataFrame(results).to_csv(partial_save_file, index=False)
    logging.info(f"Промежуточные результаты сохранены в {partial_save_file}")

def handle_exit(signum, frame):
    logging.warning("Получен сигнал завершения! Сохраняю результаты...")
    save_partial_results()
    exit(1)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

MAX_WORKERS = 10  # Оптимальное количество потоков для 500 RPM
try:
    logging.info(f"Запускаю обработку с {MAX_WORKERS} потоками и паузой {PAUSE_BETWEEN_BATCHES:.2f} секунд...")
    process_chunks_with_rate_limit(texts_for_embedding, API_KEY, max_workers=MAX_WORKERS)
finally:
    logging.info("Завершаю выполнение. Сохраняю финальные результаты...")
    save_partial_results()
    logging.info("Все результаты сохранены.")
