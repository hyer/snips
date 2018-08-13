from tools.dir_opt import Dir


def get_sym_map(map_file):
    with open(map_file, 'r') as f:
        data = f.readlines()
        syms = []

        for i in range(len(data)):
            sym = data[i].split("\n")[0]
            syms.append(sym)
    return syms

syms = get_sym_map("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/charmap_$.txt")
save_root = "/home/hyer/datasets/OCR/SSD_OCR/SSD_OCR_3/Annotations_$"

diro = Dir()

xmls = diro.getPaths("/home/hyer/datasets/OCR/SSD_OCR/SSD_OCR_3/Annotations", ".xml")
total = len(xmls)
count = 0
for xml in xmls:
    if count % 5000 == 0:
        print count
    count += 1
    # in_file = '/home/hyer/datasets/OCR/SSD_OCR/SSD_OCR_3/Annotations/000001.xml'
    with open(xml, 'r') as fin:
        data = fin.readlines()

    out_file = xml.split("/")[-1]
    with open(save_root + "/" + out_file, 'w') as fout:
        for li in data:
            line = li.strip("\n")
            if "<name>Su Huijia</name>" in line:
                fout.write(line + "\n")
                continue
            if "<name>" in line:
                label_idx = int(line.split("<name>")[-1].split("</name>")[0])
                label = syms[label_idx]
                # print label_idx, label
                new_line = "<name>{}</name>".format(label)
                # print new_line
                fout.write(new_line + "\n")
            else:
                 fout.write(line + "\n")