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

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TEXT_MODEL "Text Model"
#define KG_MODEL "KG Model"
#define JOINT_MODEL "Joint Model"

typedef float real;                    // Precision of float numbers

struct vocab_item {
    long long cn;
    char *item;
};

struct model_var {
    const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M items in the vocabulary
    char train_file[MAX_STRING], output_file[MAX_STRING];
    char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
    struct vocab_item *vocab;
    int *vocab_hash;
    long long vocab_max_size = 1000, vocab_size = 0;
    long long train_items = 0, item_count_actual = 0, file_size = 0;
    real starting_alpha;
    real *syn0, *syn1neg;
    char name[MAX_STRING];
    int *table;
}text_model, kg_model;

struct model_var2 {
    char train_file[MAX_STRING];
    long long train_items = 0, item_count_actual = 0, file_size = 0;
    real starting_alpha;
    char name[MAX_STRING];
}joint_model;

int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
long long layer1_size = 100;
long long iter = 5;
real alpha = 0.025, sample = 1e-3;
real *expTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;

void InitUnigramTable(struct model_var *model) {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    model->table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < model->vocab_size; a++) train_items_pow += pow(model->vocab[a].cn, power);
    i = 0;
    d1 = pow(model->vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        model->table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(model->vocab[i].cn, power) / train_items_pow;
        }
        if (i >= model->vocab_size) i = model->vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *item, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
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

void ReadEntity(char *item, FILE *fin) {
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
int GetItemHash(struct model_var *model,char *item) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(item); a++) hash = hash * 257 + item[a];
    hash = hash % model->vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(struct model_var *model, char *item) {
    unsigned int hash = GetItemHash(model, item);
    while (1) {
        if (model->vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, model->vocab[model->vocab_hash[hash]].item)) return model->vocab_hash[hash];
        hash = (hash + 1) % model->vocab_hash_size;
    }
    return -1;
}

// Reads a item and returns its index in the vocabulary
int ReadItemIndex(struct model_var *model, FILE *fin) {
    char item[MAX_STRING];
    if(!strcmp(TEXT_MODEL, model->name))
        ReadWord(item, fin);
    else
        ReadEntity(item, fin);
    if (feof(fin)) return -1;
    return SearchVocab(model, item);
}

// Adds a word or entity to the vocabulary
int AddItemToVocab(struct model_var *model, char *item) {
    unsigned int hash, length = strlen(item) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    model->vocab[model->vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(model->vocab[model->vocab_size].item, item);
    model->vocab[model->vocab_size].cn = 0;
    model->vocab_size++;
    // Reallocate memory if needed
    if (model->vocab_size + 2 >= model->vocab_max_size) {
        model->vocab_max_size += 1000;
        model->vocab = (struct vocab_item *)realloc(model->vocab, model->vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetItemHash(model, item);
    while (model->vocab_hash[hash] != -1) hash = (hash + 1) % model->vocab_hash_size;
    model->vocab_hash[hash] = model->vocab_size - 1;
    return model->vocab_size - 1;
}

// Used later for sorting by item counts
int VocabCompare( const void *a, const void *b) {
    return ((struct vocab_item *)b)->cn - ((struct vocab_item *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct model_var *model) {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&model->vocab[1], model->vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < model->vocab_hash_size; a++) model->vocab_hash[a] = -1;
    size = model->vocab_size;
    model->train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((model->vocab[a].cn < min_count) && (a != 0)) {
            model->vocab_size--;
            free(model->vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetItemHash(model, model->vocab[a].item);
            while (model->vocab_hash[hash] != -1) hash = (hash + 1) % model->vocab_hash_size;
            model->vocab_hash[hash] = a;
            model->train_items += model->vocab[a].cn;
        }
    }
    model->vocab = (struct vocab_item *)realloc(model->vocab, (model->vocab_size + 1) * sizeof(struct vocab_item));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct model_var *model) {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < model->vocab_size; a++) if (model->vocab[a].cn > min_reduce) {
        model->vocab[b].cn = model->vocab[a].cn;
        model->vocab[b].item = model->vocab[a].item;
        b++;
    } else free(model->vocab[a].item);
    model->vocab_size = b;
    for (a = 0; a < model->vocab_hash_size; a++) model->vocab_hash[a] = -1;
    for (a = 0; a < model->vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetItemHash(model, model->vocab[a].item);
        while (model->vocab_hash[hash] != -1) hash = (hash + 1) % model->vocab_hash_size;
        model->vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void LearnVocabFromTrainFile(struct model_var *model) {
    char item[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < model->vocab_hash_size; a++) model->vocab_hash[a] = -1;
    fin = fopen(model->train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    model->vocab_size = 0;
    AddItemToVocab(model, (char *)"</s>");
    while (1) {
        if(!strcmp(TEXT_MODEL, model->name))
            ReadWord(item, fin);
        else
            ReadEntity(item, fin);
        if (feof(fin)) break;
        model->train_items++;
        if ((debug_mode > 1) && (model->train_items % 100000 == 0)) {
            printf("%lldK%c", model->train_items / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(model, item);
        if (i == -1) {
            a = AddItemToVocab(model, item);
            model->vocab[a].cn = 1;
        } else model->vocab[i].cn++;
        if (model->vocab_size > model->vocab_hash_size * 0.7) ReduceVocab(model);
    }
    SortVocab(model);
    if (debug_mode > 0) {
        printf("%s Vocab size: %lld\n", model->name, model->vocab_size);
        printf("Items of %s in train file: %lld\n", model->name, model->train_items);
    }
    if(!strcmp(TEXT_MODEL, model->name))
        model->file_size = ftell(fin);
    else
        model->file_size = model->vocab[0].cn;
    fclose(fin);
}

void SaveVocab(struct model_var *model) {
    long long i;
    FILE *fo = fopen(model->save_vocab_file, "wb");
    for (i = 0; i < model->vocab_size; i++) fprintf(fo, "%s %lld\n", model->vocab[i].item, model->vocab[i].cn);
    fclose(fo);
}

void ReadVocab(struct model_var *model) {
    long long a, i = 0;
    char c;
    char item[MAX_STRING];
    FILE *fin = fopen(model->read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < model->vocab_hash_size; a++) model->vocab_hash[a] = -1;
    model->vocab_size = 0;
    while (1) {
        if(!strcmp(TEXT_MODEL, model->name))
            ReadWord(item, fin);
        else
            ReadEntity(item, fin);
        if (feof(fin)) break;
        a = AddItemToVocab(model, item);
        fscanf(fin, "%lld%c", &model->vocab[a].cn, &c);
        i++;
    }
    SortVocab(model);
    if (debug_mode > 0) {
        printf("%s Vocab size: %lld\n", model->name, model->vocab_size);
        printf("Items of %s in train file: %lld\n", model->name, model->train_items);
    }
    if(!strcmp(TEXT_MODEL, model->name)){
        fin = fopen(model->train_file, "rb");
        if (fin == NULL) {
            printf("ERROR: training data file not found!\n");
            exit(1);
        }
        fseek(fin, 0, SEEK_END);
        model->file_size = ftell(fin);
        fclose(fin);
    }
    else
        model->file_size = model->vocab[0].cn;
}

void InitNet(struct model_var *model) {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&(model->syn0), 128, (long long)model->vocab_size * layer1_size * sizeof(real));
    if (model->syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (negative>0) {
        a = posix_memalign((void **)&(model->syn1neg), 128, (long long)model->vocab_size * layer1_size * sizeof(real));
        if (model->syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < model->vocab_size; a++) for (b = 0; b < layer1_size; b++)
            model->syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < model->vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        model->syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}

void *TrainTextModelThread(void *id) {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(text_model.train_file, "rb");
    fseek(fi, text_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 10000) {
            text_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, text_model.name, alpha,
                       text_model.item_count_actual / (real)(text_model.train_items + 1) * 100,
                       text_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = text_model.starting_alpha * (1 - text_model.item_count_actual / (real)(iter * text_model.train_items + 1));
            if (alpha < text_model.starting_alpha * 0.0001) alpha = text_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadItemIndex(&text_model, fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
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
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > text_model.train_items / num_threads)) {
            text_model.item_count_actual += word_count - last_word_count;
            break;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        //train skip-gram
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
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
                    target = text_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += text_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * text_model.syn0[c + l1];
            }
            // Learn weights input -> hidden
            for (c = 0; c < layer1_size; c++) text_model.syn0[c + l1] += neu1e[c];
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
    long long d, entity, example, sentence_length = 0, sentence_position = 0;
    long long entity_count = 0, last_entity_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    int skip_line = 0;
    long long start_line = 0;
    ssize_t read;
    size_t len;
    char * line = NULL;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(kg_model.train_file, "rb");
    start_line = kg_model.file_size / (long long)num_threads * (long long)id;
    while(skip_line < start_line){
        skip_line++;
        if((read = getline(&line, &len, fi)) == -1)
            break;
    }
    while (1) {
        if (entity_count - last_entity_count > 10000) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            last_entity_count = entity_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  entities/thread/sec: %.2fk  ", 13, kg_model.name, alpha,
                       kg_model.item_count_actual / (real)(kg_model.file_size + 1) * 100,
                       kg_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = kg_model.starting_alpha * (1 - kg_model.item_count_actual / (real)(iter * kg_model.file_size + 1));
            if (alpha < kg_model.starting_alpha * 0.0001) alpha = kg_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while(1){
                while (1) {
                    if (sentence_length == 0)
                        entity_count ++;
                    entity = ReadItemIndex(&kg_model, fi);
                    if (feof(fi)) break;
                    if (entity == -1){
                        if(0==sentence_length){
                            getline(&line, &len, fi);
                            break;
                        }
                        else
                            continue;
                    }
                    if (entity == 0) break;
                    sen[sentence_length] = entity;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) {getline(&line, &len, fi);break;}
                }
                if( feof(fi) || sentence_length>=2)
                    break;
                else
                    sentence_length = 0;
            }
            sentence_position = 1;
        }
        if (feof(fi) || (entity_count > kg_model.file_size / num_threads)) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            break;
        }
        entity = sen[0];
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        //train skip-gram
        for (; sentence_position<sentence_length; sentence_position++){
            example = sen[sentence_position];
            if (example == -1) continue;
            l1 = example * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = entity;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = kg_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (kg_model.vocab_size - 1) + 1;
                    if (target == entity) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * kg_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
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
    long long b, d, item, last_word, sentence_length = 0, sentence_position = 0, sentence_limit=0;
    long long anchor_count = 0, last_anchor_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    char tmp_item[MAX_STRING];
    int skip_line = 0;
    long long start_line = 0;
    ssize_t read;
    size_t len;
    char * line = NULL;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(joint_model.train_file, "rb");
    start_line = joint_model.file_size / (long long)num_threads * (long long)id;
    while(skip_line < start_line){
        skip_line++;
        if((read = getline(&line, &len, fi)) == -1)
            break;
    }
    while (1) {
        if (anchor_count - last_anchor_count > 10000) {
            joint_model.item_count_actual += anchor_count - last_anchor_count;
            last_anchor_count = anchor_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  anchors/thread/sec: %.2fk  ", 13, joint_model.name, alpha,
                       joint_model.item_count_actual / (real)(joint_model.file_size + 1) * 100,
                       joint_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = joint_model.starting_alpha * (1 - joint_model.item_count_actual / (real)(iter * joint_model.file_size + 1));
            if (alpha < joint_model.starting_alpha * 0.0001) alpha = joint_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while(1){
                while (1) {
                    ReadEntity(tmp_item, fi);
                    if(sentence_length==0){
                        item = SearchVocab(&kg_model, tmp_item);
                        anchor_count++;
                    }
                    else
                        item = SearchVocab(&text_model, tmp_item);
                    if (feof(fi)) break;
                    if (item == -1){
                        if(0==sentence_length){
                            getline(&line, &len, fi);
                            break;
                        }
                        else
                            continue;
                    }
                    if (item == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sentence_length!=0 && sample > 0) {
                        real ran = (sqrt(text_model.vocab[item].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[item].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = item;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) {getline(&line, &len, fi);break;}
                }
                if( feof(fi) || sentence_length>=2)
                    break;
                else
                    sentence_length = 0;
            }
            sentence_position = 1;
        }
        if (feof(fi) || (anchor_count > joint_model.file_size / num_threads)) {
            joint_model.item_count_actual += anchor_count - last_anchor_count;
            break;
        }
        item = sen[0];
        l1 = item * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        sentence_limit = sentence_length;
        if((window+1-b)<sentence_length)sentence_limit = window+1-b;
        //train skip-gram
        for (; sentence_position < sentence_limit; sentence_position++){
            last_word = sen[sentence_position];
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
                for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * kg_model.syn0[c + l1];
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

void TrainModel(char *model_name) {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    if(!strcmp(model_name, TEXT_MODEL)){
        printf("Starting training using file %s\n", text_model.train_file);
        text_model.starting_alpha = alpha;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainTextModelThread, (void *)a);
    }
    else if(!strcmp(model_name, KG_MODEL)){
        printf("Starting training using file %s\n", kg_model.train_file);
        kg_model.starting_alpha = alpha;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainKgModelThread, (void *)a);
    }
    else{
        printf("Starting training using file %s\n", joint_model.train_file);
        joint_model.starting_alpha = alpha;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainJointModelThread, (void *)a);
    }
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
}

void SaveVector(struct model_var *model){
    FILE *fo;
    long a, b;
    fo = fopen(model->output_file, "wb");
    // Save the item vectors
    fprintf(fo, "%lld %lld\n", model->vocab_size, layer1_size);
    for (a = 0; a < model->vocab_size; a++) {
        fprintf(fo, "%s ", model->vocab[a].item);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(model->syn0[a * layer1_size + b]), sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", model->syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void InitModel(struct model_var *model){
    model->vocab = (struct vocab_item *)calloc(model->vocab_max_size, sizeof(struct vocab_item));
    model->vocab_hash = (int *)calloc(model->vocab_hash_size, sizeof(int));
    if (model->read_vocab_file[0] != 0) ReadVocab(model); else LearnVocabFromTrainFile(model);
    if (model->save_vocab_file[0] != 0) SaveVocab(model);
    if (model->output_file[0] == 0) return;
    InitNet(model);
    if (negative > 0) InitUnigramTable(model);
}

void InitJointModel(){
    joint_model.file_size = 0;
    joint_model.train_items = 0;
    
    //read anchor file to initialize the file size
    char item[MAX_STRING];
    FILE *fin;
    fin = fopen(joint_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    while (1) {
        ReadEntity(item, fin);
        if (feof(fin)) break;
        joint_model.train_items++;
        if(!strcmp(item,"</s>"))
            joint_model.file_size++;
        if ((debug_mode > 1) && (joint_model.train_items % 100000 == 0)) {
            printf("%lldK%c", joint_model.train_items / 1000, 13);
            fflush(stdout);
        }
    }
    joint_model.train_items -= joint_model.file_size;
    if (debug_mode > 0) {
        printf("%s contains %lld anchors with %lld context words\n", joint_model.name, joint_model.file_size, joint_model.train_items);
    }
    fclose(fin);
    
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
    int i, local_iter=0;
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
        printf("\t-output_word <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-output_entity <file>\n");
        printf("\t\tUse <file> to save the resulting entity vectors / entity clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word  / entity vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between (anchor) words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of (anchor) words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
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
        printf("\t-save_word_vocab <file>\n");
        printf("\t\tThe word vocabulary will be saved to <file>\n");
        printf("\t-read_word_vocab <file>\n");
        printf("\t\tThe word vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-save_entity_vocab <file>\n");
        printf("\t\tThe entity vocabulary will be saved to <file>\n");
        printf("\t-read_entity_vocab <file>\n");
        printf("\t\tThe entity vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    text_model.output_file[0] = 0;
    text_model.save_vocab_file[0] = 0;
    text_model.read_vocab_file[0] = 0;
    kg_model.output_file[0] = 0;
    kg_model.save_vocab_file[0] = 0;
    kg_model.read_vocab_file[0] = 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train_text", argc, argv)) > 0) strcpy(text_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_kg", argc, argv)) > 0) strcpy(kg_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_anchor", argc, argv)) > 0) strcpy(joint_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save_entity_vocab", argc, argv)) > 0) strcpy(kg_model.save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_entity_vocab", argc, argv)) > 0) strcpy(kg_model.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save_word_vocab", argc, argv)) > 0) strcpy(text_model.save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_word_vocab", argc, argv)) > 0) strcpy(text_model.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output_word", argc, argv)) > 0) strcpy(text_model.output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output_entity", argc, argv)) > 0) strcpy(kg_model.output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

    
    strcpy(text_model.name, (char *)TEXT_MODEL);
    strcpy(kg_model.name, (char *)KG_MODEL);
    strcpy(joint_model.name, (char *)JOINT_MODEL);
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    //read vocab & initilize text model and kg model
    InitModel(&text_model);
    InitModel(&kg_model);
    InitJointModel();
    //start training
    while(local_iter<iter){
        local_iter++;
        printf("Start training the %d time... ", local_iter);
        TrainModel((char *)TEXT_MODEL);
        TrainModel((char *)KG_MODEL);
        TrainModel((char *)JOINT_MODEL);
        printf("\niter %d success!\n", local_iter);
    }
    printf("saving results...\n");
    SaveVector(&text_model);
    SaveVector(&kg_model);
    return 0;
}
