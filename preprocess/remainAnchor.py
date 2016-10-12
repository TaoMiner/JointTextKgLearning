import nltk as nk
import codecs
import regex as re
import datetime

anchor_p = r"\[\[((?>[^\[\]]+|(?R))*)\]\]"
anchor_text_p = r'(?<=\[\[).*?(?=\]\])'
trim_href_p = r"(^.+?\t\t)|(=+References=+(.*)$)|(\[http(.*?)\])"
anchor_split_p = r'\|'
eng_p = r'[a-zA-Z]+'
num_p = r'[0-9]+'
brace_p = r'\(.*?\)'
para_p = r'={2,}'
non_num_eng = r'^[\W]+$'
p1 = r'^\'\'\'(.*)\'\'\'$'
p1_content = r'(?<=\'\'\').*?(?=\'\'\')'
p2 = r'^\"(.*)\"$'
p2_content = r'(?<=\").*?(?=\")'
p3 = r'^\'\'(.*)\'\'$'
p3_content = r'(?<=\'\').*?(?=\'\')'
outlink_split_p = r'\t\t|;'

file_input = './data/wiki2016/enwiki-text.dat'
corpus = './data/wiki2016/train_text'
# kg_corpus = './data/wiki2016/train_kg'
# kg_input = './data/wiki2016/enwiki-outlink.dat'
ent_dic_file = './data/wiki2016/ent_mention'
wiki_dic_file = './data/wiki2016/enwiki-ID.dat'
'''
file_input = '/Volumes/cyx500/enwiki/enwiki-abstract.dat'
kg_input = '/Volumes/cyx500/enwiki/enwiki-outlink.dat'
corpus = 'train_text_sample'
kg_corpus = 'train_kg'
ent_dic_file = 'ent_dic'
wiki_dic_file = '/Volumes/cyx500/enwiki/enwiki-ID.dat'
'''
ent_mention = {}
ent_dic = {}

def extractEnt(anchor_text):
    items = re.split(anchor_split_p, re.search(anchor_text_p, anchor_text).group())
    entity = items[0]
    if len(items) == 2:
        mention = items[1].strip()
        # extract three pattern mention to reduce the mention vocab
        if re.match(p1, mention):
            mention = re.search(p1_content, mention).group()
        elif re.match(p2, mention):
            mention = re.search(p2_content, mention).group()
        elif re.match(p3, mention):
            mention = re.search(p3_content, mention).group()
        mention = mention.strip()
    elif len(items) == 1:
        mention = items[0]
    else:
        # not anchor format
        return None
    #if mention is "" or " " or pure numbers or pure symbols, del the anchor
    if mention == "" or mention == " " or re.match(num_p, mention) or re.match(non_num_eng, mention):
        return None
    if entity not in ent_dic:
        #not an entity in wikipedia entity vocab, return the mention text
        mention_text = []
        segment(mention, mention_text)
        return " ".join(mention_text)
    # add entity and its mention into dict
    if entity in ent_mention:
        mention_set = ent_mention[entity]
        if mention not in mention_set:
            mention_set[mention] = 1
        else:
            mention_set[mention] += 1
    elif len(entity)>1 and len(mention)>1:
        mention_set = {}
        mention_set[mention] = 1
        ent_mention[entity] = mention_set
    else:
        # ignore the length <1 entity and mention
        return None
    if len(items) == 2:
        return "[["+entity+"|"+mention+"]]"
    else:
        return "[["+entity+"]]"

def segment(sent, words):
    tmp_words = nk.tokenize.word_tokenize(sent)
    for word in tmp_words:
        if re.match(eng_p, word):
            words.append(word)
        elif re.match(num_p, word):
            words.append("ddd")

with codecs.open(wiki_dic_file, 'r', encoding='UTF-8') as fin_id:
    for line in  fin_id:
        ents = re.split(r'\t\t', line)
        if len(ents)==2 and ents[0]!="" and ents[0]!=" ":
            ent_dic[ents[0].lower()] = ents[1].strip("\n")
    print("successfully load %d entities!" % len(ent_dic))
'''
with codecs.open(kg_input,'r', encoding='UTF-8') as fin_kg:
    with codecs.open(kg_corpus,'w', encoding='UTF-8') as fout_kg:
        line_count = 0
        for line in fin_kg:
            line_count += 1
            if line_count%10000 ==0:
                print("has processed: %d entities." % line_count)
            links = []
            line = line.lower()
            page = re.split(outlink_split_p, line)
            if len(page)<2 or len(page[0])<2 or page[0] not in ent_dic:
                continue
            for link in page[1:]:
                if link in ent_dic:
                    links.append(link)
            if len(links)<1:
                continue
            fout_kg.writelines(page[0]+"\t"+";".join(links)+"\n")
'''
with codecs.open(file_input, 'r', encoding='UTF-8') as fin:
    with codecs.open(corpus, 'w', encoding='UTF-8') as fout_text:
        line_count = 0
        texts = []
        anchors = []
        starttime = datetime.datetime.now()
        for line in fin:
            line_count += 1
            if line_count%10000 == 0 :
                endtime = datetime.datetime.now()
                print("%chas processed: %d lines, takes %d seconds..." % (13,line_count, (endtime - starttime).seconds))
            # split the paragraphs after removing references, head entity and href
            paras = re.split(para_p, re.sub(trim_href_p, "", line.lower()))
            for para in paras:
                sent_pos = 0
                words_set = []
                # skip the para within length of 30 or Nonetype
                if not para or len(para) <=30:
                    continue
                # iterate all the anchors in wiki text
                for anchor in re.finditer(anchor_p, para):
                    segment(para[sent_pos:anchor.start()], words_set)
                    anchor_word = extractEnt(anchor.group())
                    if anchor_word:
                        words_set.append(anchor_word)
                    sent_pos = anchor.end()
                if sent_pos < len(para):
                    segment(para[sent_pos:len(para)], words_set)
                if len(words_set) > 8:
                    texts.append(" ".join(words_set)+"\n")
                    if len(texts) >= 10000:
                        fout_text.writelines(texts)
                        del texts[:]
        if len(texts) > 0:
            fout_text.writelines(texts)

with codecs.open(ent_dic_file, 'w', encoding='UTF-8') as fout_ent:
    dics = []
    for ent in ent_mention:
        dics.append(ent_dic[ent]+"\t"+ent+"\t"+"\t".join(["%s=%s" % (k, v) for k, v in ent_mention[ent].items()])+"\n")
        if len(dics) >= 10000:
            fout_ent.writelines(dics)
            del dics[:]
    if len(dics) > 0:
        fout_ent.writelines(dics)
