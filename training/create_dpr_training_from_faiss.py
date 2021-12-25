import json
import re

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DPRQuestionEncoder


def clean_question(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = result.replace("[deleted]", "")
    return result.lower().strip()


def cleanup_references(text):
    # URL reference where we need to remove both the link text and URL
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal
    # views on slavery ([1](_URL_2_ & pg=PA173), [2](_URL_1_), [3](_URL_5_)).
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal views on slavery.
    result = re.sub(r"[\(\s]*\[\d+\]\([^)]+\)[,)]*", "", text, 0, re.MULTILINE)

    # URL reference where we need to preserve link text but remove URL
    # At the outbreak of the Civil War, [Leyburn left his church](_URL_19_) and joined the South.
    # At the outbreak of the Civil War, Leyburn left his church and joined the South.
    result = re.sub(r"\[([^]]+)\]\([^)]+\)", "\\1", result, 0, re.MULTILINE)

    # lastly remove just dangling _URL_[0-9]_ URL references
    result = re.sub(r"_URL_\d_", "", result, 0, re.MULTILINE)
    return result


def main():
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

    def embed_questions_for_retrieval(questions):
        query = question_tokenizer(questions, max_length=256, padding="max_length", truncation=True,
                                   return_tensors="pt")
        with torch.no_grad():
            q_reps = question_model(query["input_ids"].to(device),
                                    query["attention_mask"].to(device)).pooler_output
        return q_reps.cpu().numpy()

    def query_index(question, topk=21):
        question_embedding = embed_questions_for_retrieval([question])
        scores, wiki_passages = kilt_wikipedia_paragraphs.get_nearest_examples("embeddings", question_embedding, k=topk)

        retrieved_examples = []
        r = list(zip(wiki_passages[k] for k in columns))
        for i in range(topk):
            retrieved_examples.append({k: v for k, v in zip(columns, [r[j][0][i] for j in range(len(columns))])})

        return retrieved_examples

    def find_positive_and_hard_negative_ctxs(dataset_index: int, n_positive=1):
        positive_context_list = []
        hard_negative_context_list = []
        question = clean_question(dataset[dataset_index]["title"])
        passages = query_index(question)
        passage_list = [dict([(k, p[k]) for k in columns]) for p in passages]

        query_passage_pairs = [[question, passage["text"]] for passage in passage_list]

        features = ce_tokenizer(query_passage_pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            passage_scores = ce_model(**features).logits

        for p_idx, p in enumerate(passage_list):
            p["score"] = passage_scores[p_idx].item()

        # order by scores
        def score_passage(item):
            return item["score"]

        # pick the most relevant as the positive answer
        best_passage_list = sorted(passage_list, key=score_passage, reverse=True)
        for idx, item in enumerate(best_passage_list):
            if idx < n_positive:
                positive_context_list.append({"title": item["title"], "text": item["text"]})
            else:
                break

        # least relevant as hard_negative
        worst_passage_list = sorted(passage_list, key=score_passage, reverse=False)
        n_negatives = 7
        for idx, hard_negative in enumerate(worst_passage_list):
            if idx < n_negatives * n_positive:
                hard_negative_context_list.append({"title": hard_negative["title"], "text": hard_negative["text"]})
            else:
                break
        assert len(positive_context_list) * n_negatives == len(hard_negative_context_list)
        return positive_context_list, hard_negative_context_list

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    question_encoder_name = "vblagoje/dpr-question_encoder-single-lfqa-base"
    question_tokenizer = AutoTokenizer.from_pretrained(question_encoder_name)
    question_model = DPRQuestionEncoder.from_pretrained(question_encoder_name).to(device)
    _ = question_model.eval()

    ce_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    ce_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    _ = ce_model.eval()

    index_file_name = "../data/kilt_dpr_wikipedia.faiss"
    kilt_wikipedia = load_dataset("kilt_wikipedia", split="full")

    kilt_wikipedia_paragraphs = kilt_wikipedia.map(articles_to_paragraphs, batched=True,
                                                   remove_columns=['kilt_id', 'wikipedia_id', 'wikipedia_title', 'text',
                                                                   'anchors', 'categories', 'wikidata_info', 'history'],
                                                   batch_size=256,
                                                   cache_file_name=f"../data/wiki_kilt_paragraphs_full.arrow",
                                                   desc="Expanding wiki articles into paragraphs")

    # use paragraphs that are not simple fragments or very short sentences
    kilt_wikipedia_paragraphs = kilt_wikipedia_paragraphs.filter(
        lambda x: (x["end_character"] - x["start_character"]) > 200)
    columns = ['wikipedia_id', 'start_paragraph_id', 'start_character', 'end_paragraph_id', 'end_character',
               'title', 'section', 'text']

    kilt_wikipedia_paragraphs.load_faiss_index("embeddings", index_file_name, device=0)

    eli5_train_set = load_dataset("vblagoje/eli5v1", split="train")
    eli5_validation_set = load_dataset("vblagoje/eli5v1", split="validation")
    eli5_test_set = load_dataset("vblagoje/eli5v1", split="test")

    for dataset_name, dataset in zip(["train", "validation", "test"], [eli5_train_set,
                                                                       eli5_validation_set,
                                                                       eli5_test_set]):

        progress_bar = tqdm(range(len(dataset)), desc="Creating DPR formatted question/passage docs")
        with open('eli5-dpr-' + dataset_name + '.jsonl', 'w') as fp:
            step_size = 7
            for idx, example in enumerate(dataset):
                start = 0
                end = 7
                positive_context, hard_negative_ctxs = find_positive_and_hard_negative_ctxs(idx)
                for pc in positive_context:
                    hnc = hard_negative_ctxs[start:end]
                    json.dump({"id": example["q_id"],
                               "question": clean_question(example["title"]),
                               "positive_ctxs": pc,
                               "hard_negative_ctxs": hnc}, fp)
                    fp.write("\n")
                    start += step_size
                    end += step_size
                progress_bar.update(1)


main()
