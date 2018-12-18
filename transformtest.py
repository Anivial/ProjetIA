import io
file = io.open("test", "w", encoding="utf-8")
with io.open("data/word_phon.txt.1", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.2", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.3", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.4", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.5", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.6", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
with io.open("data/word_phon.txt.7", 'r', encoding="utf-8") as source:
    for line in source:
        line = line.strip().split("\t")
        file.write(" ".join(line[0]) + "\t" + " ".join(line[1:]) + "\n")
file.close()