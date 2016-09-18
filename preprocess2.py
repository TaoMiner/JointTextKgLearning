import codecs
import regex as re

anchor_file = './data/wiki2016/train_anchors_sample'
kg_file = './data/wiki2016/enwiki-outlink.dat'
kg_sample = './data/wiki2016/train_kg_sample'
ent_p = r'\t\t'
ent_dic = set()

with codecs.open(anchor_file, 'r', encoding='UTF-8') as fin:
    line_count = 0
    for line in fin:
        line_count += 1
	items = re.split(ent_p,line)
	if items[0]!="":
            ent_dic.add(items[0])
    print("#%d anchors." % (line_count/2))
    print("#%d unique entities." % len(ent_dic))

with codecs.open(kg_file, 'r', encoding='UTF-8') as fin_kg:
    with codecs.open(kg_sample, 'w', encoding='UTF-8') as fout_kg:
	line_count = 0
        tmp_kg = []
        for line in fin_kg:
	    tmp_line = []
	    line = line.lower()
	    line_count += 1
	    if line_count%100000 ==0:
		print("has processed: %d entities." % line_count)
            nodes = re.split(r'\t\t', line)
	    if nodes[0] in ent_dic:
		tmp_line.append(nodes[0])
		for _ in re.split(r';', nodes[1]):
                    if _!="" and _ in ent_dic:
			tmp_line.append(_)
		if len(tmp_line)>1:
		    tmp_kg.append(tmp_line[0]+"\t\t"+";".join(tmp_line[1:])+"\n")
		    if len(tmp_kg)>=10000:
                	fout_kg.writelines(tmp_kg)
                	del tmp_kg[:]
	if len(tmp_kg)>0:
	    fout_kg.writelines(tmp_kg)


