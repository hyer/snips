import glob

xml_root = "/home/hyer/data/k1_scan/Annotations"
save = "/home/hyer/data/k1_scan/Annotations_new"

paths = []
for fn in glob.glob(xml_root + "/*" + ".xml"):
    paths.append(fn)

for xml in paths:
    with open(xml, "r") as fin:
        data = fin.readlines()
    skip = True
    with open(save + "/" + xml.split("/")[-1], "w") as fout:
        for li in data:
            if skip == True:
                skip = False
                continue
            if "database" in li:
                fout.write("\t\t<database>Syms Dection</database>\n"
                            "\t\t<annotation>PASCAL VOC2007</annotation>\n"
                            "\t\t<image>flickr</image>\n"
                            "\t\t<flickrid>NULL</flickrid>\n")
                continue
            fout.write(li)
print("done")