from datasets import load_dataset

dataset = load_dataset("SimpleStories/SimpleStories")


def dataset_filter(rows):
    return [
        100 <= c <= 1000 and f <= 4
        for c, f in zip(rows["character_count"], rows["flesch_kincaid_grade"])
    ]


filtered_train = dataset["train"].filter(
    dataset_filter, batched=True, batch_size=1000, num_proc=4
)
filtered_test = dataset["test"].filter(
    dataset_filter, batched=True, batch_size=1000, num_proc=4
)

with open("train.txt", "w") as f:
    for text in filtered_train["story"]:
        f.write(text + "<END>")

with open("test.txt", "w") as f:
    for text in filtered_test["story"]:
        f.write(text + "<END>")
