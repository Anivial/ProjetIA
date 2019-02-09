import io
import random

train = io.open("data_train/shuffled_data_train.txt", "w", encoding="utf-8")
test = io.open("data_train/shuffled_data_test.txt", "w", encoding="utf-8")
tab = []
with io.open("data_train/data.txt", 'r', encoding="utf-8") as source:
    for line in source:
        tab.append(line)
random.shuffle(tab)

for i in range(0, len(tab)):
    if i < 65000:
        train.write(tab[i])
    else:
        test.write(tab[i])

train.close()
test.close()
