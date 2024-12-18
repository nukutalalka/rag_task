FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY papers.csv .
COPY partial_results.csv .
COPY embeddings_titles.npy .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.address=0.0.0.0"]
