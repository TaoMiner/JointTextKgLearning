//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 1100          //>longest entity + longest mention + 1
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TEXT_MODEL "Text Model"
#define KG_MODEL "KG Model"
#define JOINT_MODEL "Joint Model"
#define MAX_MENTION 135

typedef float real;                    // Precision of float numbers

struct anchor_item {
    long long start_pos;
    long long length;
    int entity_index;
};

struct vocab_item {
    long long vocab_mention_index;
    long long cn;
    char *item;
};

struct vocab_mention {
    int sense_num;
    long long vocab_word_index;
    char *item;
};

struct model_var3 {
    char read_vocab_file[MAX_STRING], output_file[MAX_STRING];
    long long vocab_max_size, vocab_size;
    real *syn0;
    real *mu;
    int *cluster_size;
    struct vocab_mention *vocab;
    int *vocab_hash, vocab_hash_size;
}mentions;

struct model_var {
    char train_file[MAX_STRING], output_file[MAX_STRING];
    char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
    struct vocab_item *vocab;
    int *vocab_hash, vocab_hash_size;
    long long vocab_max_size, vocab_size;
    long long train_items, item_count_actual, file_size;
    real starting_alpha, alpha;
    real *syn0, *syn1neg;
    char name[MAX_STRING];
    int *table;
}text_model, kg_model;      //for text model, syn0 is global vector, syn1neg records the only sense embedding for the words out of the mention vocab;

struct model_var2 {
    char train_file[MAX_STRING];
    long long train_items, item_count_actual, file_size, anchor_count;
    real starting_alpha, alpha;
    char name[MAX_STRING];
}joint_model;


int local_iter=0, binary = 0, debug_mode = 2, window = 5, min_count = 0, num_threads = 12, min_reduce = 1, cw = 0, sg = 1, max_sense_num = 10;
long long layer1_size = 100;
long long iter = 5, tmp_anchor_num = 0;
real alpha = 0.025, sample = 1e-3, cluster_threshold = -0.5;
real *expTable;
clock_t start;
char save_vocab_path[MAX_STRING], read_vocab_path[MAX_STRING], output_path[MAX_STRING];
int negative = 5;
const int table_size = 1e8;
const char save_interval = '\t';

void InitTextUnigramTable() {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    text_model.table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < text_model.vocab_size; a++) train_items_pow += pow(text_model.vocab[a].cn, power);
    i = 0;
    d1 = pow(text_model.vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        text_model.table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(text_model.vocab[i].cn, power) / train_items_pow;
        }
        if (i >= text_model.vocab_size) i = text_model.vocab_size - 1;
    }
}

void InitKgUnigramTable() {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    kg_model.table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < kg_model.vocab_size; a++) train_items_pow += pow(kg_model.vocab[a].cn, power);
    i = 0;
    d1 = pow(kg_model.vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        kg_model.table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(kg_model.vocab[i].cn, power) / train_items_pow;
        }
        if (i >= kg_model.vocab_size) i = kg_model.vocab_size - 1;
    }
}
//return the num of mention words
//split_pos indicates word start position in mention, range from 0 to mention length -1
int SplitMention(int *split_pos, char *mention) {
    int a = 0, b = 0, ch;
    if(mention[b]!=0){
        split_pos[a] = 0;
        a++;
        b++;
    }
    while (mention[b]!=0) {
        ch = mention[b];
        b++;
        if (ch == ' ')
            if(mention[b]!=0){
                split_pos[a] = b;
                a++;
                if(a>=MAX_MENTION){
                    printf("error! anchor's mention length is larger than %d!\n", MAX_MENTION);
                    printf("%s\n",mention);
                    break;
                }
            }
    }
    return a;
}

//return negative if that's not anchor, the int indicates how many words should be offset.
//return the word's start pos
int ReadAnchor(char *item, FILE *fin){
    int a = 0, ch, is_anchor = 0, prev_brace = 0, post_brace = 0, word_num=0, anchor_pos=0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if( prev_brace==0 ){
            if(ch == '['){
                prev_brace = 1;
                continue;
            }
            else{
                a=0;
                is_anchor --;
                break;
            }
        }
        is_anchor--;
        if(ch == ' ') word_num ++;
        if (ch == '\n' || word_num>MAX_MENTION) {
            a = 0;
            break;
        }
        if (ch == '|')
            anchor_pos = a+1;
        if ( (ch == ']') && (post_brace==0)){
            post_brace = 1;
            continue;
        }
        if(post_brace==1){
            if(ch == ']')
                is_anchor = anchor_pos;
            break;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1){printf("error! too long string:\n %s\n",item); a--;break;}
    }
    item[a] = 0;
    return is_anchor;
}


//return negative if that's not anchor, the int indicates the offset.
//return 0 if read correctly
int ReadMention(char *item, FILE *fin){
    int a = 0, ch, is_anchor = 0, prev_brace = 0, post_brace = 0, word_num=0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if( prev_brace==0 ){
            if(ch == '{'){
                prev_brace = 1;
                continue;
            }
            else{
                a=0;
                is_anchor --;
                break;
            }
        }
        is_anchor--;
        if(ch == ' ') word_num ++;
        if (ch == '\n' || word_num>MAX_MENTION) {
            a = 0;
            break;
        }
        if ( (ch == '}') && (post_brace==0)){
            post_brace = 1;
            continue;
        }
        if(post_brace==1){
            if(ch == '}')
                is_anchor = 0;
            break;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1){printf("error! too long string:\n %s\n",item); a--;break;}
    }
    item[a] = 0;
    return is_anchor;
}

// Reads a single word or anchor from a file, assuming space + tab + EOL to be word boundaries, [[anchor]]
//return -1 if read a word, 0 if [[anchor]], '|''s pos(1-length) if [[entity|mention]]
//return -2 if read a mention
int ReadText(char *item, FILE *fin) {
    int a = 0, ch, is_anchor=-1;
    
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == ']') || (ch == '[') || (ch == '{') || (ch == '}')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                if (ch == '[') ungetc(ch, fin);
                if (ch == '{') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return -1;
            }
            else if (ch == '['){
                is_anchor = ReadAnchor(item, fin);
                if (is_anchor<0){
                    fseek(fin, is_anchor, SEEK_CUR);
                    continue;
                }
                else return is_anchor;
            }
            else if(ch == '{'){
                is_anchor = ReadMention(item, fin);
                if (is_anchor<0){
                    fseek(fin, is_anchor, SEEK_CUR);
                    continue;
                }
                else return -2;
            }
            else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
    return -1;
}

void ReadItem(char *item, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ';') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return;
            } else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % text_model.vocab_hash_size;
    return hash;
}

// Returns hash value of a word
int GetMentionHash(char *mention) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(mention); a++) hash = hash * 257 + mention[a];
    hash = hash % mentions.vocab_hash_size;
    return hash;
}

// Returns hash value of an entity
int GetEntityHash(char *entity) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(entity); a++) hash = hash * 257 + entity[a];
    hash = hash % kg_model.vocab_hash_size;
    return hash;
}


// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchTextVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (text_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, text_model.vocab[text_model.vocab_hash[hash]].item)) return text_model.vocab_hash[hash];
        hash = (hash + 1) % text_model.vocab_hash_size;
    }
    return -1;
}

// Returns position of a mention in the vocabulary; if the mention is not found, returns -1
int SearchMentionVocab(char *mention) {
    unsigned int hash = GetMentionHash(mention);
    while (1) {
        if (mentions.vocab_hash[hash] == -1) return -1;
        if (!strcmp(mention, mentions.vocab[mentions.vocab_hash[hash]].item)) return mentions.vocab_hash[hash];
        hash = (hash + 1) % mentions.vocab_hash_size;
    }
    return -1;
}

// Returns position of an entity in the vocabulary; if the entity is not found, returns -1
int SearchKgVocab(char *entity) {
    unsigned int hash = GetEntityHash(entity);
    while (1) {
        if (kg_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(entity, kg_model.vocab[kg_model.vocab_hash[hash]].item)) return kg_model.vocab_hash[hash];
        hash = (hash + 1) % kg_model.vocab_hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, long long mention_index) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    text_model.vocab[text_model.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(text_model.vocab[text_model.vocab_size].item, word);
    text_model.vocab[text_model.vocab_size].cn = 0;
    text_model.vocab[text_model.vocab_size].vocab_mention_index = mention_index;
    if (mention_index>=0){
        mentions.vocab[mention_index].vocab_word_index = text_model.vocab_size;
        mentions.vocab[mention_index].sense_num = 0;
    }
    text_model.vocab_size++;
    // Reallocate memory if needed
    if (text_model.vocab_size + 2 >= text_model.vocab_max_size) {
        text_model.vocab_max_size += 100000;
        text_model.vocab = (struct vocab_item *)realloc(text_model.vocab, text_model.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetWordHash(word);
    while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
    text_model.vocab_hash[hash] = text_model.vocab_size - 1;
    return text_model.vocab_size - 1;
}

// Adds a mention to the vocabulary
int AddMentionToVocab(char *mention) {
    unsigned int hash, length = strlen(mention) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    mentions.vocab[mentions.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(mentions.vocab[mentions.vocab_size].item, mention);
    mentions.vocab[mentions.vocab_size].vocab_word_index = -1;
    mentions.vocab[mentions.vocab_size].sense_num = -1;
    mentions.vocab_size++;
    // Reallocate memory if needed
    if (mentions.vocab_size + 2 >= mentions.vocab_max_size) {
        mentions.vocab_max_size += 100000;
        mentions.vocab = (struct vocab_mention *)realloc(mentions.vocab, mentions.vocab_max_size * sizeof(struct vocab_mention));
    }
    hash = GetMentionHash(mention);
    while (mentions.vocab_hash[hash] != -1) hash = (hash + 1) % mentions.vocab_hash_size;
    mentions.vocab_hash[hash] = mentions.vocab_size - 1;
    return mentions.vocab_size - 1;
}

// Adds an entity to the vocabulary
int AddEntityToVocab(char *entity) {
    unsigned int hash, length = strlen(entity) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    kg_model.vocab[kg_model.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(kg_model.vocab[kg_model.vocab_size].item, entity);
    kg_model.vocab[kg_model.vocab_size].cn = 0;
    kg_model.vocab[kg_model.vocab_size].vocab_mention_index = -1;
    kg_model.vocab_size++;
    // Reallocate memory if needed
    if (kg_model.vocab_size + 2 >= kg_model.vocab_max_size) {
        kg_model.vocab_max_size += 100000;
        kg_model.vocab = (struct vocab_item *)realloc(kg_model.vocab, kg_model.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetEntityHash(entity);
    while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
    kg_model.vocab_hash[hash] = kg_model.vocab_size - 1;
    return kg_model.vocab_size - 1;
}

// Used later for sorting by item counts
int VocabCompare( const void *a, const void *b) {
    return ((struct vocab_item *)b)->cn - ((struct vocab_item *)a)->cn;
}

// Used later for sorting by item counts
int VocabMentionCompare( const void *a, const void *b) {
    return ((struct vocab_mention *)b)->sense_num - ((struct vocab_mention *)a)->sense_num;
}

// Sorts the text vocabulary by frequency using word counts
void SortTextVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&text_model.vocab[1], text_model.vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    size = text_model.vocab_size;
    text_model.train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((text_model.vocab[a].cn < min_count) && (a != 0)) {
            text_model.vocab_size--;
            free(text_model.vocab[a].item);
            // the corresponding mention will be set to -1, ready to be discarded from the mention vocab
            if (text_model.vocab[a].vocab_mention_index!= -1){
                mentions.vocab[text_model.vocab[a].vocab_mention_index].sense_num = -1;
                mentions.vocab[text_model.vocab[a].vocab_mention_index].vocab_word_index = -1;
            }
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(text_model.vocab[a].item);
            while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
            text_model.vocab_hash[hash] = a;
            text_model.train_items += text_model.vocab[a].cn;
        }
    }
    text_model.vocab = (struct vocab_item *)realloc(text_model.vocab, (text_model.vocab_size + 1) * sizeof(struct vocab_item));
    //re-compute the mention's vocab_word_index
    for(a=0;a<text_model.vocab_size;a++)
        if(text_model.vocab[a].vocab_mention_index!=-1)
            mentions.vocab[text_model.vocab[a].vocab_mention_index].vocab_word_index = a;
    
    qsort(&mentions.vocab[0], mentions.vocab_size, sizeof(struct vocab_mention), VocabMentionCompare);
    for (a = 0; a < mentions.vocab_hash_size; a++) mentions.vocab_hash[a] = -1;
    size = mentions.vocab_size;
    for (a = 0; a < size; a++) {
        // mention's sense_num set -1 will be discarded from the vocab
        if ((mentions.vocab[a].sense_num < 0)) {
            mentions.vocab_size--;
            free(mentions.vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetMentionHash(mentions.vocab[a].item);
            while (mentions.vocab_hash[hash] != -1) hash = (hash + 1) % mentions.vocab_hash_size;
            mentions.vocab_hash[hash] = a;
        }
    }
    mentions.vocab = (struct vocab_mention *)realloc(mentions.vocab, mentions.vocab_size * sizeof(struct vocab_mention));
    //re-compute the word's vocab_mention_index
    for(a=0;a<mentions.vocab_size;a++)
        if(mentions.vocab[a].vocab_word_index!=-1)
            text_model.vocab[mentions.vocab[a].vocab_word_index].vocab_mention_index = a;
    
}

// Sorts the vocabulary by frequency using word counts
void SortKgVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&kg_model.vocab[1], kg_model.vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    size = kg_model.vocab_size;
    kg_model.train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((kg_model.vocab[a].cn < min_count) && (a != 0)) {
            kg_model.vocab_size--;
            free(kg_model.vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetEntityHash(kg_model.vocab[a].item);
            while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
            kg_model.vocab_hash[hash] = a;
            kg_model.train_items += kg_model.vocab[a].cn;
        }
    }
    kg_model.vocab = (struct vocab_item *)realloc(kg_model.vocab, (kg_model.vocab_size + 1) * sizeof(struct vocab_item));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceTextVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < text_model.vocab_size; a++) if (text_model.vocab[a].cn > min_reduce) {
        text_model.vocab[b].cn = text_model.vocab[a].cn;
        text_model.vocab[b].item = text_model.vocab[a].item;
        b++;
    } else free(text_model.vocab[a].item);
    text_model.vocab_size = b;
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    for (a = 0; a < text_model.vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(text_model.vocab[a].item);
        while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
        text_model.vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceKgVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < kg_model.vocab_size; a++) if (kg_model.vocab[a].cn > min_reduce) {
        kg_model.vocab[b].cn = kg_model.vocab[a].cn;
        kg_model.vocab[b].item = kg_model.vocab[a].item;
        b++;
    } else free(kg_model.vocab[a].item);
    kg_model.vocab_size = b;
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    for (a = 0; a < kg_model.vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetEntityHash(kg_model.vocab[a].item);
        while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
        kg_model.vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void LearnTextVocabFromTrainFile() {
    char word[MAX_STRING], tmp_word[MAX_STRING];
    FILE *fin;
    long long a, i, b,m;
    size_t tmp_word_len = 0;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;;
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    fin = fopen(text_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    text_model.vocab_size = 0;
    AddWordToVocab((char *)"</s>", -1);
    while (1) {
        anchor_pos = ReadText(word, fin);
        if (feof(fin)) break;
        if (anchor_pos >= 0) tmp_anchor_num ++;
        
        if(anchor_pos > 0){
            tmp_word_len = strlen(word)-anchor_pos;
            strncpy(tmp_word, &word[anchor_pos], sizeof(char)*tmp_word_len);
        }
        else{
            tmp_word_len = strlen(word);
            strncpy(tmp_word, word, sizeof(char)*tmp_word_len);
        }
        tmp_word[tmp_word_len] = 0;
        m  = -1;
        words_in_mention = 1;
        if (anchor_pos != -1){
            m = SearchMentionVocab(tmp_word);
            if (m == -1) words_in_mention = SplitMention(word_begin, tmp_word);
        }
        
        for(b=0;b<words_in_mention;b++){
            if (anchor_pos == -1 || m != -1){
                tmp_word_len = strlen(tmp_word);
                strncpy(word, tmp_word, sizeof(char)*tmp_word_len);
            }
            else if(b+1<words_in_mention){
                tmp_word_len = word_begin[b+1]-1-word_begin[b];
                strncpy(word, &tmp_word[word_begin[b]], sizeof(char)*tmp_word_len);
            }
            else{
                tmp_word_len = strlen(&tmp_word[word_begin[b]]);
                strncpy(word, &tmp_word[word_begin[b]], sizeof(char)*tmp_word_len);
            }
            word[tmp_word_len]=0;
            text_model.train_items++;
            if ((debug_mode > 1) && (text_model.train_items % 100000 == 0)) {
                printf("%lldK%c", text_model.train_items / 1000, 13);
                fflush(stdout);
            }
            i = SearchTextVocab(word);
            if (i == -1) {
                a = AddWordToVocab(word, m);
                text_model.vocab[a].cn = 1;
            } else text_model.vocab[i].cn++;
            if (text_model.vocab_size > text_model.vocab_hash_size * 0.7) ReduceTextVocab();
        }
        
    }
    SortTextVocab();
    if (debug_mode > 0) {
        printf("Text Vocab size: %lld\n", text_model.vocab_size);
        printf("Words in train file: %lld\n", text_model.train_items);
        printf("Mention Vocab size after sort: %lld\n", mentions.vocab_size);
    }
    text_model.file_size = ftell(fin);
    fclose(fin);
}

void LearnKgVocabFromTrainFile() {
    char entity[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    fin = fopen(kg_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    kg_model.vocab_size = 0;
    AddEntityToVocab((char *)"</s>");
    while (1) {
        ReadItem(entity, fin);
        if (feof(fin)) break;
        
        kg_model.train_items++;
        if ((debug_mode > 1) && (kg_model.train_items % 100000 == 0)) {
            printf("%lldK%c", kg_model.train_items / 1000, 13);
            fflush(stdout);
        }
        i = SearchKgVocab(entity);
        if (i == -1) {
            a = AddEntityToVocab(entity);
            kg_model.vocab[a].cn = 1;
        } else kg_model.vocab[i].cn++;
        if (kg_model.vocab_size > kg_model.vocab_hash_size * 0.7) ReduceKgVocab();
        
    }
    SortKgVocab();
    if (debug_mode > 0) {
        printf("Entity Vocab size: %lld\n", kg_model.vocab_size);
        printf("Entity in train file: %lld\n", kg_model.train_items);
    }
    kg_model.file_size = ftell(fin);
    fclose(fin);
}

void SaveMentionVocab() {
    long long i;
    FILE *fo = fopen("./mv", "wb");
    for (i = 0; i < mentions.vocab_size; i++) fprintf(fo, "%s\t%lld\n", mentions.vocab[i].item, mentions.vocab[i].vocab_word_index);
    fclose(fo);
}

void SaveKgVocab() {
    long long i;
    FILE *fo = fopen(kg_model.save_vocab_file, "wb");
    for (i = 0; i < kg_model.vocab_size; i++) fprintf(fo, "%s\t%lld\n", kg_model.vocab[i].item, kg_model.vocab[i].cn);
    fclose(fo);
}

void SaveTextVocab() {
    long long i;
    FILE *fo = fopen(text_model.save_vocab_file, "wb");
    for (i = 0; i < text_model.vocab_size; i++) fprintf(fo, "%s\t%lld\n", text_model.vocab[i].item, text_model.vocab[i].cn);
    fclose(fo);
}

void ReadMentionVocab(){
    long long a, i = 0, b;
    char c;
    char mention[MAX_STRING];
    FILE *fin = fopen(mentions.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Mention Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < mentions.vocab_hash_size; a++) mentions.vocab_hash[a] = -1;
    mentions.vocab_size = 0;
    while (1) {
        ReadItem(mention, fin);
        if (feof(fin)) break;
        a = AddMentionToVocab(mention);
        fscanf(fin, "%lld%c", &b, &c);
        i++;
    }
    if (debug_mode > 0) {
        printf("Mention vocab size: %lld\n", mentions.vocab_size);
    }
    fclose(fin);
}

void ReadTextVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(text_model.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Text Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    text_model.vocab_size = 0;
    while (1) {
        ReadItem(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word, SearchMentionVocab(word));
        fscanf(fin, "%lld%c", &text_model.vocab[a].cn, &c);
        i++;
    }
    SortTextVocab();
    if (debug_mode > 0) {
        printf("Text vocab size: %lld\n", text_model.vocab_size);
        printf("Words in train file: %lld\n", text_model.train_items);
        printf("Mention Vocab size after sort: %lld\n", mentions.vocab_size);
    }
    fin = fopen(text_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    text_model.file_size = ftell(fin);
    fclose(fin);
}

void ReadKgVocab() {
    long long a, i = 0;
    char c;
    char entity[MAX_STRING];
    FILE *fin = fopen(kg_model.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    kg_model.vocab_size = 0;
    while (1) {
        ReadItem(entity, fin);
        if (feof(fin)) break;
        a = AddEntityToVocab(entity);
        fscanf(fin, "%lld%c", &kg_model.vocab[a].cn, &c);
        i++;
    }
    SortKgVocab();
    if (debug_mode > 0) {
        printf("Entity Vocab size: %lld\n", kg_model.vocab_size);
        printf("entity in train file: %lld\n", kg_model.train_items);
    }
    fin = fopen(kg_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    kg_model.file_size = ftell(fin);
    fclose(fin);
}

void InitTextNet() {
    long long a, b, c;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&(text_model.syn0), 128, (long long)text_model.vocab_size * layer1_size * sizeof(real));
    if (text_model.syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (negative>0) {
        a = posix_memalign((void **)&(text_model.syn1neg), 128, (long long)text_model.vocab_size * layer1_size * sizeof(real));
        if (text_model.syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < text_model.vocab_size; a++) for (b = 0; b < layer1_size; b++)
            text_model.syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < text_model.vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        text_model.syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
    
    a = posix_memalign((void **)&(mentions.syn0), 128, (long long)mentions.vocab_size * max_sense_num * layer1_size * sizeof(real));
    if (mentions.syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&(mentions.mu), 128, (long long)mentions.vocab_size * max_sense_num * layer1_size * sizeof(real));
    if (mentions.mu == NULL) {printf("Memory allocation failed\n"); exit(1);}
    mentions.cluster_size = (int *)calloc(mentions.vocab_size*max_sense_num, sizeof(int));
    for (a = 0; a < mentions.vocab_size; a++)
        for (b = 0; b < max_sense_num;b++)
            for (c = 0; c < layer1_size; c++) {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                mentions.syn0[a * max_sense_num * layer1_size + b*layer1_size + c] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
            }
    for (a = 0; a < mentions.vocab_size; a++)
        for (b = 0; b < max_sense_num;b++)
            for (c = 0; c < layer1_size; c++)
                mentions.mu[a * max_sense_num * layer1_size + b*layer1_size + c] = 0;
}

void InitKgNet() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&(kg_model.syn0), 128, (long long)kg_model.vocab_size * layer1_size * sizeof(real));
    if (kg_model.syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (negative>0) {
        a = posix_memalign((void **)&(kg_model.syn1neg), 128, (long long)kg_model.vocab_size * layer1_size * sizeof(real));
        if (kg_model.syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < kg_model.vocab_size; a++) for (b = 0; b < layer1_size; b++)
            kg_model.syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < kg_model.vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        kg_model.syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}


int findMentionSense(long long mention_index, real *vec){
    int sense_index = -1, a, b, closest = -1, c_size;
    real dist = 0.0, closest_dist = -1.0, len_v, len_mu;
    if (mentions.vocab[mention_index].sense_num > 0){
        len_v = 0;
        for (a = 0; a < layer1_size; a++) len_v += vec[a] * vec[a];
        len_v = sqrt(len_v);
        for (a = 0; a < mentions.vocab[mention_index].sense_num; a++){
            c_size = mentions.cluster_size[mention_index*max_sense_num+a];
            if(c_size<=0)
                continue;
            len_mu = 0;
            for (b = 0; b < layer1_size; b++)
                len_mu += (mentions.mu[mention_index * max_sense_num * layer1_size + a * layer1_size + b]/c_size) * (mentions.mu[mention_index * max_sense_num * layer1_size + a * layer1_size + b]/c_size);
            len_mu = sqrt(len_mu);
            dist = 0;
            for (b = 0; b < layer1_size; b++)
                dist += (vec[b]/len_v) * (mentions.mu[mention_index * max_sense_num * layer1_size + a * layer1_size + b]/c_size/len_mu);
            if(dist > closest_dist){
                closest_dist = dist;
                closest = a;
            }
        }
    }
    if (closest_dist < cluster_threshold){
        if(mentions.vocab[mention_index].sense_num >= max_sense_num) return -1;
        //create new
        sense_index = mentions.vocab[mention_index].sense_num;
        mentions.vocab[mention_index].sense_num++;
    }
    else{
        sense_index = closest;
    }
    //update the cluster mu, and cluster_size
    mentions.cluster_size[mention_index*max_sense_num+sense_index] ++;
    for(b=0;b<layer1_size;b++)
        mentions.mu[mention_index * max_sense_num * layer1_size + sense_index * layer1_size + b] += vec[b];
    return sense_index;
}

void *TrainTextModelThread(void *id) {
    long long a, b, d, m, word=-1, sense_index=-1, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, context_num = 0;
    unsigned long long next_random = (long long)id;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    char item[MAX_STRING], tmp_item[MAX_STRING], str_word[MAX_STRING];
    size_t tmp_item_len = 0;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    real *context_vec = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(text_model.train_file, "rb");
    fseek(fi, text_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    while (1) {
        if (word_count - last_word_count > 10000) {
            text_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, text_model.name, text_model.alpha,
                       text_model.item_count_actual / (real)(text_model.train_items + 1) * 100,
                       text_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            text_model.alpha = text_model.starting_alpha * (1 - text_model.item_count_actual / (real)(iter * text_model.train_items + 1));
            if (text_model.alpha < text_model.starting_alpha * 0.0001) text_model.alpha = text_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                anchor_pos = ReadText(item, fi);
                if (feof(fi)) break;
                
                if(anchor_pos > 0){
                    tmp_item_len = strlen(item)-anchor_pos;
                    strncpy(tmp_item, &item[anchor_pos], sizeof(char)*tmp_item_len);
                }
                else{
                    tmp_item_len = strlen(item);
                    strncpy(tmp_item, item, sizeof(char)*tmp_item_len);
                }
                tmp_item[tmp_item_len] = 0;
                m  = -1;
                words_in_mention = 1;
                if (anchor_pos != -1){
                    m = SearchMentionVocab(tmp_item);
                    if (m == -1) words_in_mention = SplitMention(word_begin, tmp_item);
                }
                
                for(b=0;b<words_in_mention;b++){
                    if (anchor_pos == -1 || m != -1){
                        tmp_item_len = strlen(tmp_item);
                        strncpy(str_word, tmp_item, sizeof(char)*tmp_item_len);
                    }
                    else if(b+1<words_in_mention){
                        tmp_item_len = word_begin[b+1]-1-word_begin[b];
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    else{
                        tmp_item_len = strlen(&tmp_item[word_begin[b]]);
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    str_word[tmp_item_len]=0;
                    word = SearchTextVocab(str_word);
                    word_count++;
                    if (word == -1) continue;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ran = (sqrt(text_model.vocab[word].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                if(word == 0 || sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > text_model.train_items / num_threads)) {
            text_model.item_count_actual += word_count - last_word_count;
            break;
        }
        
        word = sen[sentence_position];
        if (word == -1) continue;
        
        sense_index = -1;
        //if word is a mention
        if (text_model.vocab[word].vocab_mention_index!=-1){
            for (c = 0; c < layer1_size; c++) context_vec[c] = 0;
            context_num = 0;
            for (a = 0; a < window * 2 + 1; a++) if (a != window) {
                d = sentence_position - window + a;
                if (d < 0) continue;
                if (d >= sentence_length) break;
                for (c = 0; c < layer1_size; c++)
                    context_vec[c] += text_model.syn0[d*layer1_size+c];
                context_num ++;
            }
            if (context_num>0){
                for (c = 0; c < layer1_size; c++)
                    context_vec[c] /= context_num;
                sense_index = findMentionSense(text_model.vocab[word].vocab_mention_index, context_vec);
                if(sense_index>=max_sense_num || sense_index < 0) sense_index = -1;
            }
        }
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        if (sense_index == -1) l1 = word * layer1_size;
        else l1 = text_model.vocab[word].vocab_mention_index * max_sense_num * layer1_size + sense_index * layer1_size;
        //train skip-gram
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = last_word;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = text_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                    if (target == last_word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                if (sense_index == -1){
                    for (c = 0; c < layer1_size; c++) f += text_model.syn1neg[c + l1] * text_model.syn0[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * text_model.alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * text_model.alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * text_model.alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn0[c + l2];
                    for (c = 0; c < layer1_size; c++) text_model.syn0[c + l2] += g * text_model.syn1neg[c + l1];
                }
                else{
                    for (c = 0; c < layer1_size; c++) f += mentions.syn0[c + l1] * text_model.syn0[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * text_model.alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * text_model.alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * text_model.alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn0[c + l2];
                    for (c = 0; c < layer1_size; c++) text_model.syn0[c + l2] += g * mentions.syn0[c + l1];
                }
            }
            // Learn weights input -> hidden
            if (sense_index == -1)
                for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l1] += neu1e[c];
            else
                for (c = 0; c < layer1_size; c++) mentions.syn0[c + l1] += neu1e[c];
        }
        
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void *TrainKgModelThread(void *id) {
    long long d, entity, head_entity = -1, line_entity_count = 0, is_read_head = 1, sentence_length = 0, sentence_position = 0;
    long long entity_count = 0, last_entity_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    char tmp_item[MAX_STRING];
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(kg_model.train_file, "rb");
    fseek(fi, kg_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    if((long long)id!=0)
        while(1){
            ReadItem(tmp_item, fi);
            entity = SearchKgVocab(tmp_item);
            entity_count ++;
            if (feof(fi) || (entity==0)) break;
        }
    head_entity = -1;
    is_read_head = 1;
    line_entity_count = 0;
    while (1) {
        if (entity_count - last_entity_count > 10000) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            last_entity_count = entity_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  entities/thread/sec: %.2fk  ", 13, kg_model.name, kg_model.alpha,
                       kg_model.item_count_actual / (real)(kg_model.train_items + 1) * 100,
                       kg_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            kg_model.alpha = kg_model.starting_alpha * (1 - kg_model.item_count_actual / (real)(iter * kg_model.train_items + 1));
            if (kg_model.alpha < kg_model.starting_alpha * 0.0001) kg_model.alpha = kg_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while(1){
                ReadItem(tmp_item, fi);
                if (feof(fi)) break;
                entity_count ++;
                line_entity_count++;
                if(is_read_head==1){
                    head_entity = SearchKgVocab(tmp_item);
                    if(head_entity==0) line_entity_count = 0;
                    if(head_entity>0 && line_entity_count==1) is_read_head=0;
                    else head_entity = -1;
                    continue;
                }
                else if(head_entity!=-1){
                    entity = SearchKgVocab(tmp_item);
                    if (entity == -1) continue;
                    if (entity == 0) {
                        line_entity_count = 0;
                        is_read_head=1;
                        break;
                    }
                }
                else {is_read_head=1;continue;}
                sen[sentence_length] = entity;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (entity_count > kg_model.train_items / num_threads)) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            break;
        }
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        //train skip-gram
        for (; sentence_position<sentence_length; sentence_position++){
            entity = sen[sentence_position];
            if (entity == -1) continue;
            l1 = entity * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = head_entity;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = kg_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (kg_model.vocab_size - 1) + 1;
                    if (target == head_entity) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * kg_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * kg_model.alpha;
                else if (f < -MAX_EXP) g = (label - 0) * kg_model.alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * kg_model.alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * kg_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) kg_model.syn1neg[c + l2] += g * kg_model.syn0[c + l1];
            }
            // Learn weights input -> hidden
            for (c = 0; c < layer1_size; c++) kg_model.syn0[c + l1] += neu1e[c];
        }
        sentence_length = 0;
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

//use entity to predict word
void *TrainJointModelThread(void *id) {
    long long a, b, d, m, word=-1, last_word, sentence_length = 0, sentence_position = 0, anchor_position = 0;
    long long word_count = 0, anchor_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    char tmp_item[MAX_STRING], item[MAX_STRING], str_word[MAX_STRING];
    size_t tmp_item_len = 0;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    real f, g;
    clock_t now;
    struct anchor_item *anchors = (struct anchor_item *)calloc(MAX_SENTENCE_LENGTH, sizeof(struct anchor_item));
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(joint_model.train_file, "rb");
    fseek(fi, joint_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    while (1) {
        if (word_count - last_word_count > 10000) {
            joint_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, joint_model.name, joint_model.alpha,
                       joint_model.item_count_actual / (real)(joint_model.train_items + 1) * 100,
                       joint_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            joint_model.alpha = joint_model.starting_alpha * (1 - joint_model.item_count_actual / (real)(iter * joint_model.train_items + 1));
            if (joint_model.alpha < joint_model.starting_alpha * 0.0001) joint_model.alpha = joint_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            anchor_count = 0;
            while(1){
                anchor_pos = ReadText(item, fi);
                if (feof(fi)) break;
                if (anchor_pos>=0) anchors[anchor_count].start_pos = sentence_length;
                if(anchor_pos > 0){
                    tmp_item_len = strlen(item)-anchor_pos;
                    strncpy(tmp_item, &item[anchor_pos], sizeof(char)*tmp_item_len);
                }
                else{
                    tmp_item_len = strlen(item);
                    strncpy(tmp_item, item, sizeof(char)*tmp_item_len);
                }
                tmp_item[tmp_item_len] = 0;
                m  = -1;
                words_in_mention = 1;
                if (anchor_pos != -1){
                    m = SearchMentionVocab(tmp_item);
                    if (m == -1) words_in_mention = SplitMention(word_begin, tmp_item);
                }
                
                for(b=0;b<words_in_mention;b++){
                    if (anchor_pos == -1 || m != -1){
                        tmp_item_len = strlen(tmp_item);
                        strncpy(str_word, tmp_item, sizeof(char)*tmp_item_len);
                    }
                    else if(b+1<words_in_mention){
                        tmp_item_len = word_begin[b+1]-1-word_begin[b];
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    else{
                        tmp_item_len = strlen(&tmp_item[word_begin[b]]);
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    str_word[tmp_item_len]=0;
                    word = SearchTextVocab(str_word);
                    word_count++;
                    if (word == -1) continue;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ran = (sqrt(text_model.vocab[word].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                if(anchor_pos>=0 && b >= words_in_mention-1){
                    anchors[anchor_count].length = sentence_length - anchors[anchor_count].start_pos;
                    if(anchors[anchor_count].length>0){
                        if (anchor_pos==0)
                            tmp_item_len = strlen(item);
                        else
                            tmp_item_len = anchor_pos-1;
                        strncpy(tmp_item, item, tmp_item_len);
                        tmp_item[tmp_item_len] = 0;
                        anchors[anchor_count].entity_index = SearchKgVocab(tmp_item);
                        if(anchors[anchor_count].entity_index!=-1){
                            anchor_count++;
                            if(anchor_count >= MAX_SENTENCE_LENGTH) anchor_count--;
                        }
                    }
                }
                if(word == 0 || sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > joint_model.train_items / num_threads)) {
            joint_model.item_count_actual += word_count - last_word_count;
            break;
        }
        
        if(anchor_count<=0){
            sentence_length = 0;
            continue;
        }
        if(cw){
            for(anchor_position=0;anchor_position<anchor_count;anchor_position++){
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                b = next_random % window;
                sentence_position = anchors[anchor_position].start_pos;
                word = anchors[anchor_position].entity_index;
                if (word == -1) continue;
                //train skip-gram
                for (a = b; a < window * 2 + 1 - b; a++)
                    if(a == window)
                        sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                    else {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        l1 = last_word * layer1_size;
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        // NEGATIVE SAMPLING
                        if (negative > 0) for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = kg_model.table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (kg_model.vocab_size - 1) + 1;
                                if (target == word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += text_model.syn0[c + l1] * kg_model.syn1neg[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * joint_model.alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * joint_model.alpha;
                            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * joint_model.alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * kg_model.syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++) kg_model.syn1neg[c + l2] += g * text_model.syn0[c + l1];
                        }
                        // Learn weights input -> hidden
                        for (c = 0; c < layer1_size; c++) text_model.syn0[c + l1] += neu1e[c];
                    }
            }
        }
        if(sg){
            for(anchor_position=0;anchor_position<anchor_count;anchor_position++){
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                b = next_random % window;
                sentence_position = anchors[anchor_position].start_pos;
                l1 = anchors[anchor_position].entity_index * layer1_size;
                //train skip-gram
                for (a = b; a < window * 2 + 1 - b; a++)
                    if(a == window)
                        sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                    else {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        // NEGATIVE SAMPLING
                        if (negative > 0) for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = last_word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = text_model.table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                                if (target == last_word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * text_model.syn0[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * joint_model.alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * joint_model.alpha;
                            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * joint_model.alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn0[c + l2];
                            for (c = 0; c < layer1_size; c++) text_model.syn0[c + l2] += g * kg_model.syn0[c + l1];
                        }
                        // Learn weights input -> hidden
                        for (c = 0; c < layer1_size; c++) kg_model.syn0[c + l1] += neu1e[c];
                    }
            }
        }
        sentence_length = 0;
    }
    fclose(fi);
    free(anchors);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainTextModel(){
    long a, b, c;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training text model using file %s\n", text_model.train_file);
    text_model.item_count_actual = 0;
    text_model.starting_alpha = text_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainTextModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    //update mention cluster for next iter
    for (a = 0; a < mentions.vocab_size; a++)
        for(b = 0;b<mentions.vocab[a].sense_num;b++){
            if(mentions.cluster_size[a*max_sense_num+b] == 0){printf("error! zero cluster size: %s:%ld!\n",mentions.vocab[a].item, b);continue;}
            for(c=0;c<layer1_size;c++)
                mentions.mu[a*max_sense_num*layer1_size+b*layer1_size+c] /= mentions.cluster_size[a*max_sense_num+b];
            mentions.cluster_size[a*max_sense_num+b] = 1;
        }
}

void TrainKgModel(){
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training using file %s\n", kg_model.train_file);
    kg_model.item_count_actual = 0;
    kg_model.starting_alpha = kg_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainKgModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
}

void TrainJointModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training using file %s\n", joint_model.train_file);
    joint_model.item_count_actual = 0;
    joint_model.starting_alpha = joint_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainJointModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
}

void SaveWordVector(int id, char *postfix){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b;
    sprintf(tmp_output_file, "%s%d%s", text_model.output_file, id, postfix);
    fo = fopen(tmp_output_file, "wb");
    // Save the item vectors
    fprintf(fo, "%lld %lld\n", text_model.vocab_size, layer1_size);
    for (a = 0; a < text_model.vocab_size; a++) {
        fprintf(fo, "%s%c", text_model.vocab[a].item, save_interval);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(text_model.syn0[a * layer1_size + b]), sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", text_model.syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void SaveMentionVector(int id, char *postfix){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b, c;
    sprintf(tmp_output_file, "%s%d%s", mentions.output_file, id, postfix);
    fo = fopen(tmp_output_file, "wb");
    // Save the item vectors
    fprintf(fo, "%lld %lld %d\n", mentions.vocab_size, layer1_size, max_sense_num);
    for (a = 0; a < mentions.vocab_size; a++) {
        fprintf(fo, "%s%c%d%c", mentions.vocab[a].item, save_interval, mentions.vocab[a].sense_num, save_interval);
        if (binary){
            for (b = 0; b < mentions.vocab[a].sense_num; b++){
                for (c = 0; c < layer1_size; c++)
                    fwrite(&(mentions.syn0[a *max_sense_num* layer1_size + b * layer1_size + c]), sizeof(real), 1, fo);
                for (c = 0; c < layer1_size; c++)
                    fwrite(&(mentions.mu[a *max_sense_num* layer1_size + b * layer1_size + c]), sizeof(real), 1, fo);
            }
            fprintf(fo, "\n");
        }
        else{
            fprintf(fo, "\n");
            for (b = 0; b < mentions.vocab[a].sense_num; b++){
                for (c = 0; c < layer1_size; c++)
                    fprintf(fo, "%lf ", mentions.syn0[a *max_sense_num* layer1_size + b * layer1_size + c]);
                fprintf(fo, "\n");
                for (c = 0; c < layer1_size; c++)
                    fprintf(fo, "%lf ", mentions.mu[a *max_sense_num* layer1_size + b * layer1_size + c]);
                fprintf(fo, "\n");
            }
        }
    }
    fclose(fo);
}

void SaveEntityVector(int id, char *postfix){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b;
    sprintf(tmp_output_file, "%s%d%s", kg_model.output_file, id, postfix);
    fo = fopen(tmp_output_file, "wb");
    // Save the item vectors
    fprintf(fo, "%lld %lld\n", kg_model.vocab_size, layer1_size);
    for (a = 0; a < kg_model.vocab_size; a++) {
        fprintf(fo, "%s%c", kg_model.vocab[a].item, save_interval);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(kg_model.syn0[a * layer1_size + b]), sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", kg_model.syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void SaveVector(int id, char *postfix){
    SaveWordVector(id, postfix);
    SaveMentionVector(id, postfix);
    SaveEntityVector(id, postfix);
}

void InitTextModel(){
    text_model.vocab_max_size = 2500000;      //vocab word size is 2.7m
    text_model.vocab_size = 0;
    text_model.train_items = 0;
    text_model.item_count_actual = 0;
    text_model.file_size = 0;
    text_model.alpha = alpha;
    text_model.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    
    text_model.vocab = (struct vocab_item *)calloc(text_model.vocab_max_size, sizeof(struct vocab_item));
    text_model.vocab_hash = (int *)calloc(text_model.vocab_hash_size, sizeof(int));
    
    mentions.vocab_size = 0;
    mentions.vocab_max_size = 1000000;      //vocab mention size is 1m
    mentions.vocab_hash_size = 5000000;  // Maximum items in the vocabulary 5m*0.7=3.5m
    mentions.vocab = (struct vocab_mention *)calloc(mentions.vocab_max_size, sizeof(struct vocab_mention));
    mentions.vocab_hash = (int *)calloc(mentions.vocab_hash_size, sizeof(int));
    
    if (mentions.read_vocab_file[0] != 0) ReadMentionVocab();
    
    if (read_vocab_path[0] != 0) ReadTextVocab();
    else LearnTextVocabFromTrainFile();
    if (save_vocab_path[0] != 0) SaveTextVocab();
    if (output_path[0] == 0) return;
    InitTextNet();
    if (negative > 0) InitTextUnigramTable();
}

void InitKgModel(){
    min_count = 1;
    kg_model.vocab_max_size = 5000000;      //vocab entity size is 5.13m
    kg_model.vocab_size = 0;
    kg_model.train_items = 0;
    kg_model.item_count_actual = 0;
    kg_model.file_size = 0;
    kg_model.alpha = alpha;
    kg_model.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    kg_model.vocab = (struct vocab_item *)calloc(kg_model.vocab_max_size, sizeof(struct vocab_item));
    kg_model.vocab_hash = (int *)calloc(kg_model.vocab_hash_size, sizeof(int));
    if (read_vocab_path[0] != 0) ReadKgVocab(); else LearnKgVocabFromTrainFile();
    if (save_vocab_path[0] != 0) SaveKgVocab();
    if (output_path[0] == 0) return;
    InitKgNet();
    if (negative > 0) InitKgUnigramTable();
}

void InitJointModel(){
    joint_model.file_size = text_model.file_size;
    joint_model.train_items = text_model.train_items;
    joint_model.item_count_actual = 0;
    joint_model.alpha = alpha;
    joint_model.anchor_count = tmp_anchor_num;
    printf("Total anchors in train file: %lld\n", joint_model.anchor_count);
}


int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Joint word&entity VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train_text <file>\n");
        printf("\t\tUse text data from <file> to train the text model\n");
        printf("\t-train_kg <file>\n");
        printf("\t\tUse knowledge data from <file> to train the knowledge model\n");
        printf("\t-train_anchor <file>\n");
        printf("\t\tUse anchor data from <file> to train align model\n");
        printf("\t-output_path <path>\n");
        printf("\t\tUse <path> to save the resulting vectors \n");
        printf("\t-read_vocab_path <path>\n");
        printf("\t\tUse <path> to read the word and entity vocab\n");
        printf("\t-save_vocab_path <path>\n");
        printf("\t\tUse <path> to save the word and entity vocab\n");
        printf("\t-read_mention_vocab <file>\n");
        printf("\t\tUse <file> to read the mention vocab\n");
        printf("\t-cluster_threshold <float>\n");
        printf("\t\tSet the vector's cosine similairty with a cluster larger than <float>, it will be assigned to the cluster; default is -0.5\n");
        printf("\t-max_sense_num <int>\n");
        printf("\t\tSet max sense number of a word; default is 10\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word  / entity vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between (anchor) words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of (anchor) words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-cw <int>\n");
        printf("\t\talign word vector to entity; default is 0\n");
        printf("\t-sg <int>\n");
        printf("\t\talign entity vector to word; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    
    read_vocab_path[0]=0;
    save_vocab_path[0]=0;
    output_path[0]=0;
    mentions.read_vocab_file[0]= 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train_text", argc, argv)) > 0) strcpy(text_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_kg", argc, argv)) > 0) strcpy(kg_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_anchor", argc, argv)) > 0) strcpy(joint_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-output_path", argc, argv)) > 0) strcpy(output_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_vocab_path", argc, argv)) > 0) strcpy(read_vocab_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-save_vocab_path", argc, argv)) > 0) strcpy(save_vocab_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_mention_vocab", argc, argv)) > 0) strcpy(mentions.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-cw", argc, argv)) > 0) cw = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sg", argc, argv)) > 0) sg = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cluster_threshold", argc, argv)) > 0) cluster_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-max_sense_num", argc, argv)) > 0) max_sense_num = atoi(argv[i + 1]);
    
    if(save_vocab_path[0]!=0){
        sprintf(text_model.save_vocab_file, "%svocab_word.txt", save_vocab_path);
        sprintf(kg_model.save_vocab_file, "%svocab_entity.txt", save_vocab_path);
    }
    if(output_path[0]!=0){
        sprintf(text_model.output_file, "%svectors_word", output_path);
        sprintf(kg_model.output_file, "%svectors_entity", output_path);
        sprintf(mentions.output_file, "%svectors_mention", output_path);
    }
    if(read_vocab_path[0]!=0){
        sprintf(text_model.read_vocab_file, "%svocab_word.txt", read_vocab_path);
        sprintf(kg_model.read_vocab_file, "%svocab_entity.txt", read_vocab_path);
    }
    
    strcpy(text_model.name, (char *)TEXT_MODEL);
    strcpy(kg_model.name, (char *)KG_MODEL);
    strcpy(joint_model.name, (char *)JOINT_MODEL);
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    //read vocab & initilize text model and kg model
    InitTextModel();
    InitKgModel();
    InitJointModel();
    //start training
    local_iter = 0;
    while(local_iter<iter){
        local_iter++;
        printf("Start training the %d time... ", local_iter);
        TrainTextModel();
        TrainKgModel();
        TrainJointModel();
        printf("\niter %d success!\n", local_iter);
        if(local_iter%1==0){
            printf("saving results...\n");
            SaveVector(local_iter, ".dat");
        }
    }
    return 0;
}
