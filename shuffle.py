import io
import random
file = io.open("data/shuffled_data.txt", "w", encoding="utf-8")
tab = []
with io.open("data/data.txt", 'r', encoding="utf-8") as source:
    for line in source:
        tab.append(line)
random.shuffle(tab)

for line in tab:
    file.write(line)
