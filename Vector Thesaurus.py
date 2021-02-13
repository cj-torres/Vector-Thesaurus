import numpy
from scipy import spatial

with open("GloVe/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
    embeddings_dict = {}
    for line in f:
        values = line.split()
        word = " ".join(values[:-300])
        vector = numpy.asarray(values[-300:], "float32")
        embeddings_dict[word] = vector


def find_mean_embeddings(words):
    try:
        vecs = numpy.stack([embeddings_dict[word] for word in words], axis=1)

    except:
        print("One or more words is not present in dictionary")
        pass

    mean_vec = numpy.mean(vecs, axis=1)
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], mean_vec))


test = ["big", "huge", "awesome", "amazing"]
sorted = find_mean_embeddings(test)
for i in sorted[0:9]:
    print(i)
