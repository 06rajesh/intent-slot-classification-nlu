from utils import import_data


def process_sentences(sentences, labels):
    limit = len(sentences)

    data = list()
    # sequences = list()
    intents = list()
    max_len = 0
    for i in range(limit):
        # for word in sentences[i]:
        #     sequences.append(word)
        intent = labels[i].intent
        tags = labels[i].positions
        entry = (sentences[i], intent, tags)
        data.append(entry)
        if intent not in  intents:
            intents.append(intent)
        if len(sentences[i]) > max_len:
            max_len = len(sentences[i])
    # vocab = list(set(sequences))
    return data, intents, max_len

if __name__ == '__main__':
    sentences, custom_labels, classes = import_data("nlu_traindev/train.json", limit=-1)

    class_to_ix = {c: i + 1 for i, c in enumerate(classes)}
    class_to_ix[""] = 0
    print(class_to_ix)

    dataset, intent_list, max_len = process_sentences(sentences, custom_labels)
    intent_to_ix = {c: i for i, c in enumerate(intent_list)}
    print(intent_to_ix)
    print(max_len)
    print(dataset[0])