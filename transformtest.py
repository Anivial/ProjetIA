import io
with io.open("data/word_phon.txt.1", 'r', encoding="utf-8") as source:

    file = io.open("test", "w", encoding="utf-8")

    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")

    file.close()