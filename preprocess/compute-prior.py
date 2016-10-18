import regex as re
import codecs

mention_count_file = 'mention_count'
ent_mention_file = 'ent_mention'
ent_prior = 'ent_prob'

mention_count = {}
ent_mention = {}

def loadEntityDic():
    with codecs.open(ent_mention_file, 'r', encoding='UTF-8') as fin_id:
        for line in fin_id:
            ents = re.split(r'\t', line.strip())
            if len(ents)>2:
                if ents[1] in ent_mention:
                    ment_cnt_dic = ent_mention[ents[1]]
                else:
                    ment_cnt_dic = {}
                    ent_mention[ents[1]] = ment_cnt_dic
                for tmp_mention in ents[2:]:
                    a_ment_cnt = re.split(r'=(?=[\d]+)', tmp_mention)
                    if len(a_ment_cnt) ==2:
                        ment_cnt_dic[a_ment_cnt[0]] = int(a_ment_cnt[1])
        print("successfully loaded %d entities!" % len(ent_mention))

def loadMentionCount():
    with codecs.open(mention_count_file, 'r', encoding='UTF-8') as fin_id:
        for line in fin_id:
            ment_cnt = re.split(r'=(?=[\d]+)', line.strip())
            if len(ment_cnt)==2 and ment_cnt[0] not in mention_count:
                mention_count[ment_cnt[0]] = int(ment_cnt[1])
            else:
                print "error with %s" % line
        print("successfully loaded %d mentions!" % len(mention_count))

loadEntityDic()
loadMentionCount()
ent_prob = {}
with codecs.open(ent_prior, 'w', encoding='UTF-8') as fout_prob:
    for ent in ent_mention:
        ent_anchor_count = 0
        ent_mention_count = 0
        ent_mention_num = 0
        for t_ment in ent_mention[ent]:
            ent_anchor_count += ent_mention[ent][t_ment]
            ent_mention_num += 1
            if t_ment in mention_count:
                ent_mention_count += mention_count[t_ment]
        if ent_anchor_count+ent_mention_count >0:
            ent_prob[ent] = [ent_mention_num, (float(ent_anchor_count)/float(ent_anchor_count+ent_mention_count))]
    ent_prob = sorted(ent_prob.iteritems(), key=lambda d:d[1][1], reverse = True)
    for ent in ent_prob:
        fout_prob.write("%s\t%f\t%d\n" % (ent[0], ent[1][1], ent[1][0]))