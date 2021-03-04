import pickle

def save_obj(obj, name):
    with open('../../obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('../../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    d = load_obj("gu-lexicon")
    '''Space Index is 1 and Blank index is 0'''
    s = set()
    s.add(("#", " ", "#"))
    for key in d:
        word = ["#"] + d[key] + ["#"]
        for index in range(1, len(word) - 1, 1):
            s.add((word[index - 1], word[index], word[index + 1]))
    triphones_to_num_labels = dict()
    num_labels_to_triphones = dict()
    label = 1
    for i in sorted(s):
        triphones_to_num_labels[i] = label
        num_labels_to_triphones[label] = i
        label += 1
    triphones_to_num_labels["_"] = 0
    num_labels_to_triphones[0] = "_"
    # print(triphones_to_num_labels)
    # exit()
    word_to_triphone = {}
    word_to_triphone_label_int = {}
    for key in d:
        triphones = list()
        triphones_labels = list()
        word = ["#"] + d[key] + ["#"]
        for index in range(1, len(word) - 1, 1):
            triphone =(word[index - 1], word[index], word[index + 1])
            triphones.append(triphone)
            triphones_labels.append(triphones_to_num_labels[triphone])

        word_to_triphone[key] = triphones
        word_to_triphone_label_int[key] = triphones_labels
    word_to_triphone_label_int[" "] = [1]
    print(len(s))
    print(s)
    count = 0
    for i in word_to_triphone:
        if count > 20:
            break
        print(i)
        print(word_to_triphone[i])
        print(i, word_to_triphone_label_int[i])
        print(d[i])
        count += 1
    save_obj(word_to_triphone_label_int, "gu-word-to-triphone-label-int")
    save_obj(triphones_to_num_labels, "gu-triphones-to-num-labels")
    save_obj(num_labels_to_triphones, "gu-num-labels-to-triphones")
    # print(d)