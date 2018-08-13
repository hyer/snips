
more_file = "/home/hyer/workspace/algo/FlowerR/data/cls_label_more.txt"
name_file = "/home/hyer/workspace/algo/FlowerR/data/cls_label.txt"
with open(name_file, "r") as f:
    data = f.readlines()

label_name_chi = []
for li in data:
    line = li.strip().split()
    chi_name = line[2]
    label_name_chi.append(chi_name)


with open(more_file, "r") as f:
    data_more = f.readlines()

label_name_chi_more = []
for li in data_more:
    line = li.strip().split()
    chi_name = line[0]
    label_name_chi_more.append(chi_name)

more_labels = []
with open("/home/hyer/workspace/algo/FlowerR/data/new_label.txt", "w") as fout:
    for lab in label_name_chi_more:
        if lab not in label_name_chi:
            more_labels.append(lab)
            fout.write(lab + "\n")

print more_labels


