import random
import json
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, cos_sim
from tqdm.auto import tqdm
from datasets import load_dataset


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


def clean_answer(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = re.sub(r"BULLET::::-", "", result)
    return trim(result.strip())


def trim(text, word_count: int = 100):
    return " ".join(text.split(" ")[:word_count])


def find_hard_negative_ctxs(dataset, dataset_embeddings, embedding_index: int,
                            exclude_answer_patterns, similarity_threshold=[0.5, 0.6], k=25, min_count=3):
    hard_negative_ctxs = []
    results = semantic_search(dataset_embeddings[embedding_index], dataset_embeddings, top_k=k,
                              score_function=cos_sim)
    # list if dicts
    # [{'corpus_id': 8, 'score': -0.019427383318543434},
    #  ...
    # {'corpus_id': 10, 'score': -0.09040290117263794}]
    # hard negative are most similar and negatives are most disimilar to embedding_index
    hard_negative_results = results[0][1:k + 1]
    assert len(hard_negative_results) > min_count * 2
    for r in hard_negative_results:
        example = dataset[r["corpus_id"]]
        if similarity_threshold[0] < r["score"] <= similarity_threshold[1]:
            for a in example["answers"]["text"]:
                hard_negative_ctxs.append({"title": "", "text": clean_answer(a)})
        if len(hard_negative_ctxs) > min_count:
            break
    return hard_negative_ctxs[:min_count]


def find_negative_ctxs(dataset, dataset_embeddings, embedding_index: int,
                       exclude_answer_patterns, similarity_threshold=0.1, k=7, min_count=3):
    negative_ctxs = []
    random_sample = random.sample(range(len(dataset_embeddings)), k * 20)
    similarities = cos_sim(dataset_embeddings[embedding_index], dataset_embeddings[random_sample])[0].tolist()
    for idx, score in enumerate(similarities):
        if score < similarity_threshold:
            example = dataset[random_sample[idx]]
            for a in example["answers"]["text"]:
                negative_ctxs.append({"title": "", "text": clean_answer(a)})
        if len(negative_ctxs) > min_count:
            break
    return negative_ctxs[:min_count]


def main():
    embedder = SentenceTransformer('all-mpnet-base-v2')

    eli5_train_set = load_dataset("vblagoje/eli5v1", split="train")
    eli5_validation_set = load_dataset("vblagoje/eli5v1", split="validation")
    eli5_test_set = load_dataset("vblagoje/eli5v1", split="test")

    train_set = embedder.encode([example["title"] for example in eli5_train_set], convert_to_tensor=True,
                                show_progress_bar=True)
    validation_set = embedder.encode([example["title"] for example in eli5_validation_set], convert_to_tensor=True,
                                     show_progress_bar=True)

    test_set = embedder.encode([example["title"] for example in eli5_test_set], convert_to_tensor=True,
                               show_progress_bar=True)
    exclude_answer_patterns = [re.compile("not sure what you"), re.compile("\n\n >")]
    for dataset_name, dataset, dataset_embeddings in zip(["train", "validation", "test"],
                                                         [eli5_train_set, eli5_validation_set, eli5_test_set],
                                                         [train_set, validation_set, test_set]):
        min_elements = 3
        skip_count = 0
        progress_bar = tqdm(range(len(dataset)), desc="Creating DPR formatted question/passage docs")
        with open('eli5-dpr-' + dataset_name + '.jsonl', 'w') as fp:
            for idx, example in enumerate(dataset):
                negative_ctxs = find_negative_ctxs(dataset, dataset_embeddings, idx, exclude_answer_patterns)
                hard_negative_ctxs = find_hard_negative_ctxs(dataset, dataset_embeddings, idx, exclude_answer_patterns)
                positive_context = [{"text": clean_answer(a), "title": ""} for a in example["answers"]["text"] if
                                    not any([p.search(a) for p in exclude_answer_patterns])]
                if not positive_context:
                    positive_context = [{"text": clean_answer(a), "title": ""} for a in example["answers"]["text"]]
                if len(positive_context) > 0 and len(negative_ctxs) > 0 and len(hard_negative_ctxs) >= min_elements:
                    json.dump({"id": example["q_id"],
                               "question": clean_question(example["title"]),
                               "positive_ctxs": positive_context[:min_elements],
                               "negative_ctxs": negative_ctxs[:min_elements],
                               "hard_negative_ctxs": hard_negative_ctxs[:min_elements]}, fp)
                    fp.write("\n")
                else:
                    skip_count += 1
                progress_bar.update(1)

        print(f"Skipped {skip_count} questions")


main()
