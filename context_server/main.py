import torch
from fastapi import FastAPI, Depends
from transformers import AutoTokenizer, AutoModel

from datasets import load_from_disk

from auth import JWTBearer

device = ("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('vblagoje/retribert-base-uncased')
model = AutoModel.from_pretrained('vblagoje/retribert-base-uncased').to(device)
_ = model.eval()

index_file_name = "./data/kilt_wikipedia.faiss"
columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
           'wikidata_info', 'history']

min_snippet_length = 20
topk = 21


kilt_wikipedia_paragraphs = load_from_disk("./data/kilt_wiki_prepared")
# use paragraphs that are not simple fragments or very short sentences
kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(lambda x: x["end_character"] > 250)
kilt_wikipedia_paragraphs.load_faiss_index("embeddings", index_file_name, device=0)


def embed_questions_for_retrieval(questions):
    query = tokenizer(questions, max_length=128, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        q_reps = model.embed_questions(query["input_ids"].to(device),
                                       query["attention_mask"].to(device)).cpu().type(torch.float)
    return q_reps.numpy()


def query_index(question):
    question_embedding = embed_questions_for_retrieval([question])
    scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)
    columns = ['wikipedia_id', 'title', 'text', 'section', 'start_paragraph_id', 'end_paragraph_id',
               'start_character', 'end_character']
    retrieved_examples = []
    r = list(zip(wiki_passages[k] for k in columns))
    for i in range(topk):
        retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})
    return retrieved_examples


app = FastAPI()


@app.get("/")
def hello():
  return {"Hello world": 1}


@app.get("/find_context",  dependencies=[Depends(JWTBearer())])
def find_context(question: str = None):
    return [res for res in query_index(question) if len(res["text"].split()) > min_snippet_length][:int(topk / 3)]



