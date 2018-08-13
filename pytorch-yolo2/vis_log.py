import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Hack', size=9)

log_file = "log/yolo2-416x416-crohme-10x-e=30.log"

with open(log_file, "r") as fin:
    data = fin.readlines()

sam_idxs = []
loss_x = []
loss_y = []
loss_w = []
loss_h = []
confs = []
clss = []
totals = []
test_epochs = []
train_epochs = []
precisions =[]
recalls = []
fscores = []
lrs = []

for li in data:
    line = li.strip("\n")
    # if "test epoch" in line:
    #     epochs


    idx = line.split(":")[0]
    try:
        sample_index = int(idx)
        if isinstance(sample_index, int):
            # print line
            sam_idxs.append(sample_index)
            loss_x.append(float(line.split(",")[3].split(" ")[-1]))
            loss_y.append(float(line.split(",")[4].split(" ")[-1]))
            loss_w.append(float(line.split(",")[5].split(" ")[-1]))
            loss_h.append(float(line.split(",")[6].split(" ")[-1]))
            confs.append(float(line.split(",")[7].split(" ")[-1]))
            clss.append(float(line.split(",")[8].split(" ")[-1]))
            totals.append(float(line.split(",")[9].split(" ")[-1]))

    except:
        pass

# print loss_x
fig0 = plt.figure('loss-sam_idxs')
plt.plot(sam_idxs, loss_x, label="loss_x")
plt.plot(sam_idxs, loss_y, label="loss_y")
plt.plot(sam_idxs, loss_w, label="loss_w")
plt.plot(sam_idxs, loss_h, label="loss_h")
plt.plot(sam_idxs, confs, label="confs")
plt.plot(sam_idxs, clss, label="clss")
plt.plot(sam_idxs, totals, '-r', label="totals")

plt.legend(loc='upper left')
# plt.show()


for li in data:
    line = li.strip("\n")
    if "test epoch" in line:
        test_epochs.append(line.split(":")[-1])

    if "precision" in line:
        precisions.append(line.split(" ")[3].split(",")[0])
        recalls.append(line.split(" ")[5].split(",")[0])
        fscores.append(line.split(" ")[7])

    if "lr" in line:
        train_epochs.append(line.split(" ")[3].split(",")[0])
        lrs.append(line.split(" ")[-1])

# print loss_x
fig1 = plt.figure('AP-test_epoch')
plt.plot(test_epochs, precisions, label="precision")
plt.plot(test_epochs, recalls, label="recall")
plt.plot(test_epochs, fscores, label="fscore")
plt.legend(loc='upper left')
print test_epochs, precisions
print test_epochs, recalls
print test_epochs, fscores

#
fig2 = plt.figure('lr')
plt.plot(train_epochs, lrs, label="lr")


plt.legend(loc='upper left')
plt.show()
