import os
import faiss
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
import torch
from datasets import load_dataset, Dataset


def main():
    dims = 128
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    eli5 = load_dataset("vblagoje/eli5v1")

    ctx_encoder_name = "vblagoje/dpr-ctx_encoder-single-lfqa-base"
    question_encoder_name = "vblagoje/dpr-question_encoder-single-lfqa-base"
    ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_encoder_name)
    ctx_model = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device)
    _ = ctx_model.eval()

    question_tokenizer = AutoTokenizer.from_pretrained(question_encoder_name)
    question_model = DPRQuestionEncoder.from_pretrained(question_encoder_name).to(device)
    _ = question_model.eval()

    index_file_name = "../data/kilt_dpr_wikipedia.faiss"
    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")
    kilt_wikipedia_columns = ['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories',
                              'wikidata_info', 'history']

    def articles_to_paragraphs(examples):
        ids, titles, sections, texts, start_ps, end_ps, start_cs, end_cs = [], [], [], [], [], [], [], []
        for bidx, example in enumerate(examples["text"]):
            last_section = ""
            for idx, p in enumerate(example["paragraph"]):
                if "Section::::" in p:
                    last_section = p
                ids.append(examples["wikipedia_id"][bidx])
                titles.append(examples["wikipedia_title"][bidx])
                sections.append(last_section)
                texts.append(p)
                start_ps.append(idx)
                end_ps.append(idx)
                start_cs.append(0)
                end_cs.append(len(p))

        return {"wikipedia_id": ids, "title": titles,
                "section": sections, "text": texts,
                "start_paragraph_id": start_ps, "end_paragraph_id": end_ps,
                "start_character": start_cs,
                "end_character": end_cs
                }

    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=kilt_wikipedia_columns,
                                                   batch_size=256,
                                                   cache_file_name=f"../data/wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(
        lambda x: (x["end_character"] - x["start_character"]) > 200)
    columns = ['wikipedia_id', 'start_paragraph_id', 'start_character', 'end_paragraph_id', 'end_character',
               'title', 'section', 'text']

    def embed_questions_for_retrieval(questions):
        query = question_tokenizer(questions, max_length=128, padding="max_length", truncation=True,
                                   return_tensors="pt")
        with torch.no_grad():
            q_reps = question_model(query["input_ids"].to(device),
                                    query["attention_mask"].to(device)).pooler_output
        return q_reps.cpu().numpy()

    def query_index(question, topk=7):
        topk = topk * 3  # grab 3x results and filter for word count
        topk = 3
        question_embedding = embed_questions_for_retrieval([question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)

        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})

        return retrieved_examples

    def create_kilt_datapoint(eli5_example, wiki_passages, min_length=20, topk=7):
        res_list = [dict([(k, p[k]) for k in columns]) for p in wiki_passages]
        res_list = [res for res in res_list if len(res["text"].split()) > min_length][:topk]

        # make a KILT data point
        # see https://github.com/facebookresearch/KILT#kilt-data-format
        output = []
        for a in eli5_example["answers"]["text"]:
            output.append({"answer": a})

        output.append({"provenance": [
            # evidence set for the answer from the KILT ks
            {
                "wikipedia_id": r["wikipedia_id"],  # *mandatory*
                "title": r["title"],
                "section": r["section"],
                "start_paragraph_id": r["start_paragraph_id"],
                "start_character": r["start_character"],
                "end_paragraph_id": r["end_paragraph_id"],
                "end_character": r["end_character"],
                "text": r["text"],
                "bleu_score": None,  # wrt original evidence
                "meta": None  # dataset/task specific
            } for r in res_list
        ]})
        return {"id": eli5_example["q_id"],
                "input": eli5_example["title"],
                "output": output,  # each element is an answer or provenance (can have multiple of each)
                "meta": None  # dataset/task specific
                }

    def create_support_doc(dataset: Dataset, output_filename: str):
        progress_bar = tqdm(range(len(dataset)), desc="Creating supporting docs")

        with open(output_filename, "w") as fp:
            for example in dataset:
                wiki_passages = query_index(example["title"])
                kilt_dp = create_kilt_datapoint(example, wiki_passages)
                json.dump(kilt_dp, fp)
                fp.write("\n")
                progress_bar.update(1)

    if not os.path.isfile(index_file_name):
        def embed_passages_for_retrieval(examples, max_length=128):
            p = ctx_tokenizer(examples["text"], max_length=max_length, padding="max_length",
                              truncation=True, return_tensors="pt")
            with torch.no_grad():
                a_reps = ctx_model(p["input_ids"].to(device),
                                   p["attention_mask"].to(device)).pooler_output
            return {"embeddings": a_reps.cpu().numpy()}

        kilt_wikipedia_paragraphs_embeddings = kilt_wikipedia_paragraphs.map(embed_passages_for_retrieval,
                                                                             batched=True, batch_size=512,
                                                                             cache_file_name="../data/kilt_embedded.arrow",
                                                                             desc="Creating faiss index")

        kilt_wikipedia_paragraphs_embeddings.add_faiss_index(column="embeddings", custom_index=faiss.IndexFlatIP(dims))
        kilt_wikipedia_paragraphs_embeddings.save_faiss_index("embeddings", index_file_name)

    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", index_file_name, device=0)
    create_support_doc(eli5["train"], "eli5_dpr_train_precomputed_dense_docs.json")
    create_support_doc(eli5["validation"], "eli5_dpr_validation_precomputed_dense_docs.json")


main()
