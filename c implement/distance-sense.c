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
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define MAX_STRING 1100

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const char split_pattern = '\t';
long long vocab_hash_size = 30000000;
int has_mention = 0;
int max_entity_size = 10;
int max_label_size = 3653051;
int size=0;

struct vocab_mention{
    char read_file[MAX_STRING];
    char *vocab;
    int *sense_num;
    long long vocab_size, layer_size;
    float *M;
    float *mu;
    int max_sense_num;
    int *vocab_hash;
}mention;

struct vocab_word{
    char read_file[MAX_STRING];
    char *vocab;
    long long vocab_size, layer_size;
    float *M;
    int *vocab_hash;
    int *mention_index;
}word;

struct vocab_entity{
    char read_file[MAX_STRING];
    char *vocab;
    long long vocab_size, layer_size;
    float *M;
}entity;

struct vocab_label{
    char *item;
    int entity_size;
    int *entity_index;
};

struct {
    struct vocab_label *vocab;
    int *vocab_hash;
    long long vocab_size;
}label;

// Returns hash value of a word
int GetItemHash(char *item) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(item); a++) hash = hash * 257 + item[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of an entity in the vocabulary; if the item is not found, returns -1
int SearchLabelVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (label.vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, label.vocab[label.vocab_hash[hash]].item)) return label.vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}


int SearchEntityVocab(char *item){
    int a, b=-1, c,i;
    b = SearchLabelVocab(item);
    if(b==-1) return -1;
    if(label.vocab[b].entity_size>1){
        printf("Entity candidates :\n");
        for(i=0;i<label.vocab[b].entity_size;i++){
            a =label.vocab[b].entity_index[i];
            if(a==-1)
                continue;
            printf("%d : %s\n",i, &entity.vocab[a*MAX_STRING]);
        }
        
        printf("please input the entity number:");
        scanf("%d",&c);
        getchar();
        b =label.vocab[b].entity_index[c];
    }
    else if(label.vocab[b].entity_size>0)
        b=label.vocab[b].entity_index[0];
    return b;
}

int AddLabelToVocab(char *item){
    unsigned int length = strlen(item) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    label.vocab[label.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(label.vocab[label.vocab_size].item, item);
    label.vocab[label.vocab_size].entity_size = 0;
    label.vocab[label.vocab_size].entity_index = (int *)calloc(max_entity_size, sizeof(int));
    label.vocab_size ++;
    
    //re alloc the memory
    if( label.vocab_size +2 >max_label_size){
        max_label_size += 10000;
        label.vocab = (struct vocab_label *)realloc(label.vocab, max_label_size * sizeof(struct vocab_label));
    }
    
    unsigned int hash = GetItemHash(item);
    while (label.vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    label.vocab_hash[hash] = label.vocab_size-1;
    return label.vocab_size-1;
}

// Adds a label to the vocabulary, return -1 if the entity exist
int AddEntityToVocab(int vocab_index) {
    char item[MAX_STRING];
    char c;
    int a, b, d;
    for(a = 0;a<MAX_STRING;a++){
        c = entity.vocab[vocab_index * MAX_STRING+a];
        if(c == 0)
            break;
        if(c == '('){
            a--;
            break;
        }
        item[a] = c;
    }
    item[a] = 0;
    b = SearchLabelVocab(item);
    if(b==-1)
        b = AddLabelToVocab(item);
    for (a=0;a<label.vocab[b].entity_size;a++)
        if(!strcmp(&entity.vocab[label.vocab[b].entity_index[a]], &entity.vocab[vocab_index]))
            return -1;
    label.vocab[b].entity_index[label.vocab[b].entity_size] = vocab_index;
    label.vocab[b].entity_size ++;
    //re alloc the memory
    if( label.vocab[b].entity_size % max_entity_size == 8){
        d = ((int)(label.vocab[b].entity_size/max_entity_size)+1)*max_entity_size;
        label.vocab[b].entity_index = (int *)realloc(label.vocab[b].entity_index, d * sizeof(int));
        for(a = label.vocab[b].entity_size;a < d;a++)
            label.vocab[b].entity_index[a] = -1;
    }
    return b;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchWordVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (word.vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, &word.vocab[word.vocab_hash[hash]*MAX_STRING])) return word.vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Returns position of a mention in the vocabulary; if the mention is not found, returns -1
int SearchMentionVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (mention.vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, &mention.vocab[mention.vocab_hash[hash]])) return mention.vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Adds a word to its hash and build its relation with mention
void AddWordToVocab(long long word_index) {
    unsigned int hash = GetItemHash(&word.vocab[word_index * MAX_STRING]);
    int mention_index = -1;
    while (word.vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    word.vocab_hash[hash] = word_index;
    if(has_mention){
        mention_index = SearchMentionVocab(&word.vocab[word_index * MAX_STRING]);
        if(mention_index!=-1) word.mention_index[word_index] = mention_index;
    }
}

// Adds a mention to its hash
void AddMentionToVocab(long long mention_index) {
    unsigned int hash = GetItemHash(&mention.vocab[mention_index * MAX_STRING]);
    while (mention.vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    mention.vocab_hash[hash] = mention_index;
}

void FindNearest(int top_n, float *vec){
    char *bestw[top_n];
    float dist, bestd[top_n];
    long long a, b, c, d;
    for (a = 0; a < top_n; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    for (a = 0; a < top_n; a++) bestd[a] = 0;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    
    //normalization
//    len = 0;
//    for (a = 0; a < size; a++) len += vec[a] * vec[a];
//    len = sqrt(len);
//    for (a = 0; a < size; a++) vec[a] /= len;
    //compute nearest words and entities
    
    printf("\n                                              word      Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < top_n; a++) bestd[a] = -1;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    //compute distance with each word
    for (c = 0; c < word.vocab_size; c++) {
        a = 0;
        //skip self
        //if (index == c) continue;
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * word.M[a + c * size];
        for (a = 0; a < top_n; a++) {
            if (dist > bestd[a]) {
                for (d = top_n - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &word.vocab[c * MAX_STRING]);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    
    printf("\n                                              mention      Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < top_n; a++) bestd[a] = -1;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    //compute distance with each word
    for (c = 0; c < mention.vocab_size; c++) {
        a = 0;
        //skip self
        //if (index == c) continue;
        dist = 0;
        for (b=0;b<mention.sense_num[c];b++){
            for (a = 0; a < size; a++) dist += vec[a] * mention.M[a + b * size + c * mention.max_sense_num * size];
            for (a = 0; a < top_n; a++) {
                if (dist > bestd[a]) {
                    for (d = top_n - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &mention.vocab[c * MAX_STRING]);
                    break;
                }
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    
    printf("\n                                              entity       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < top_n; a++) bestd[a] = -1;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    //compute distance with each word
    for (c = 0; c < entity.vocab_size; c++) {
        a = 0;
        //skip self
        //if (index == c) continue;
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * entity.M[a + c * size];
        for (a = 0; a < top_n; a++) {
            if (dist > bestd[a]) {
                for (d = top_n - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &entity.vocab[c * MAX_STRING]);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
}


void GetItem(char *item){
    long long a=0;
    printf("Enter word or entity (EXIT to break): ");
    a = 0;
    while (1) {
        item[a] = fgetc(stdin);
        if ((item[a] == '\n') || (a >= max_size - 1)) {
            item[a] = 0;
            break;
        }
        a++;
    }
}

void ReadMentionVector(){
    FILE *f;
    long long a, b, d;
    float len;
    char c;
    f = fopen(mention.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &mention.vocab_size);
    fscanf(f, "%lld", &mention.layer_size);
    fscanf(f, "%d", &mention.max_sense_num);
    mention.vocab = (char *)malloc((long long)mention.vocab_size * MAX_STRING * sizeof(char));
    mention.sense_num = (int *)malloc((long long)mention.vocab_size * sizeof(int));
    for (a = 0; a < mention.vocab_size; a++) mention.sense_num[a] = -1;
    mention.vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for (a = 0; a < vocab_hash_size; a++) mention.vocab_hash[a] = -1;
    mention.M = (float *)malloc((long long)mention.vocab_size * mention.max_sense_num * (long long)mention.layer_size * sizeof(float));
    if (mention.M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)mention.vocab_size * mention.max_sense_num * mention.layer_size * sizeof(float) / 1048576, mention.vocab_size, mention.layer_size);
        return;
    }
    mention.mu = (float *)malloc((long long)mention.vocab_size * mention.max_sense_num * (long long)mention.layer_size * sizeof(float));
    if (mention.mu == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)mention.vocab_size * mention.max_sense_num * mention.layer_size * sizeof(float) / 1048576, mention.vocab_size, mention.layer_size);
        return;
    }
    for (b = 0; b < mention.vocab_size; b++) {
        a = 0;
        while (1) {
            mention.vocab[b * MAX_STRING + a] = fgetc(f);
            if (feof(f) || (mention.vocab[b * MAX_STRING + a] == split_pattern)) break;
            if ((a < MAX_STRING) && (mention.vocab[b * MAX_STRING + a] != '\n')) a++;
        }
        mention.vocab[b * MAX_STRING + a] = 0;
        fscanf(f, "%d%c", &mention.sense_num[b],&c);
        if(mention.sense_num[b]>mention.max_sense_num) {printf("error!wrong sense number of mention %s!\n",&mention.vocab[b * MAX_STRING + a]);continue;}
        AddMentionToVocab(b);
        //read sense vec and cluster vec
        for (d = 0;d<mention.sense_num[b];d++){
            //read and normalize the d sense vector of b word
            for (a = 0; a < mention.layer_size; a++) fread(&mention.M[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size], sizeof(float), 1, f);
            len = 0;
            for (a = 0; a < mention.layer_size; a++) len += mention.M[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size] * mention.M[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size];
            len = sqrt(len);
            for (a = 0; a < mention.layer_size; a++) mention.M[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size] /= len;
            //read and normalize the d cluster vector of b word
            for (a = 0; a < mention.layer_size; a++) fread(&mention.mu[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size], sizeof(float), 1, f);
            len = 0;
            for (a = 0; a < mention.layer_size; a++) len += mention.mu[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size] * mention.mu[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size];
            len = sqrt(len);
            for (a = 0; a < mention.layer_size; a++) mention.mu[a + d * mention.layer_size + b * mention.max_sense_num * mention.layer_size] /= len;
        }
    }
    fclose(f);
}

void ReadWordVector(){
    FILE *f;
    long long a, b;
    float len;
    f = fopen(word.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &word.vocab_size);
    fscanf(f, "%lld", &word.layer_size);
    word.vocab = (char *)malloc((long long)word.vocab_size * MAX_STRING * sizeof(char));
    word.mention_index = (int *)malloc((long long)word.vocab_size * sizeof(int));
    for (a = 0; a < word.vocab_size; a++) word.mention_index[a] = -1;
    word.vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for (a = 0; a < vocab_hash_size; a++) word.vocab_hash[a] = -1;
    word.M = (float *)malloc((long long)word.vocab_size * (long long)word.layer_size * sizeof(float));
    if (word.M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)word.vocab_size * word.layer_size * sizeof(float) / 1048576, word.vocab_size, word.layer_size);
        return;
    }
    for (b = 0; b < word.vocab_size; b++) {
        a = 0;
        while (1) {
            word.vocab[b * MAX_STRING + a] = fgetc(f);
            if (feof(f) || (word.vocab[b * MAX_STRING + a] == split_pattern)) break;
            if ((a < MAX_STRING) && (word.vocab[b * MAX_STRING + a] != '\n')) a++;
        }
        word.vocab[b * MAX_STRING + a] = 0;
        AddWordToVocab(b);
        for (a = 0; a < word.layer_size; a++) fread(&word.M[a + b * word.layer_size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < word.layer_size; a++) len += word.M[a + b * word.layer_size] * word.M[a + b * word.layer_size];
        len = sqrt(len);
        for (a = 0; a < word.layer_size; a++) word.M[a + b * word.layer_size] /= len;
    }
    fclose(f);
}

void ReadEntityVector(){
    FILE *f;
    long long a, b, c;
    float len;
    f = fopen(entity.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &entity.vocab_size);
    fscanf(f, "%lld", &entity.layer_size);
    label.vocab = (struct vocab_label*)calloc(max_label_size, sizeof(struct vocab_label));
    label.vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for(a=0;a<vocab_hash_size;a++) label.vocab_hash[a] = -1;
    label.vocab_size = 0;
    entity.vocab = (char *)malloc((long long)entity.vocab_size * MAX_STRING * sizeof(char));
    entity.M = (float *)malloc((long long)entity.vocab_size * (long long)entity.layer_size * sizeof(float));
    if (entity.M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)entity.vocab_size * entity.layer_size * sizeof(float) / 1048576, entity.vocab_size, entity.layer_size);
        return;
    }
    for (b = 0; b < entity.vocab_size; b++) {
        a = 0;
        while (1) {
            entity.vocab[b * MAX_STRING + a] = fgetc(f);
            if (feof(f) || (entity.vocab[b * MAX_STRING + a] == split_pattern)) break;
            if ((a < MAX_STRING) && (entity.vocab[b * MAX_STRING + a] != '\n')) a++;
        }
        entity.vocab[b * MAX_STRING + a] = 0;
        c = AddEntityToVocab(b);
        if (c==-1){
            entity.vocab_size--;
            b--;
            continue;
        }
        for (a = 0; a < entity.layer_size; a++) fread(&entity.M[a + b * entity.layer_size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < entity.layer_size; a++) len += entity.M[a + b * entity.layer_size] * entity.M[a + b * entity.layer_size];
        len = sqrt(len);
        for (a = 0; a < entity.layer_size; a++) entity.M[a + b * entity.layer_size] /= len;
    }
    fclose(f);
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
    char item[max_size];
    int i, word_index = -1, entity_index = -1;
    long long a;
    float word_vec[max_size], entity_vec[max_size];
    int has_word = 0, has_entity = 0;
    if (argc < 2) {
        printf("\t-read_word_vector <file>\n");
        printf("\t\tUse <file> to read the resulting word vectors\n");
        printf("\t-read_entity_vector <file>\n");
        printf("\t\tUse <file> to read the resulting entity vectors\n");
        printf("\t-read_mention_vector <file>\n");
        printf("\t\tUse <file> to read the resulting mention vectors\n");
        printf("\nExamples:\n");
        printf("./distance -read_word_vector ./vec_word read_entity_vector ./vec_entity -read_mention_vector ./vec_mention\n\n");
        return 0;
    }
    word.read_file[0] = 0;
    entity.read_file[0] = 0;
    mention.read_file[0] = 0;
    word.vocab_size = 0;
    entity.vocab_size = 0;
    mention.vocab_size = 0;
    word.layer_size = 0;
    entity.layer_size = 0;
    mention.layer_size = 0;
    if ((i = ArgPos((char *)"-read_word_vector", argc, argv)) > 0) strcpy(word.read_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_entity_vector", argc, argv)) > 0) strcpy(entity.read_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_mention_vector", argc, argv)) > 0) strcpy(mention.read_file, argv[i + 1]);
    
    if(0!=mention.read_file[0]){
        printf("loading mention vectors...\n");
        ReadMentionVector();
        if(mention.vocab_size>0){
            printf("Successfully load %lld mentions with %lld dimentions from %s\n", mention.vocab_size, mention.layer_size, mention.read_file);
            has_mention = 1;
        }
    }
    
    if(0!=word.read_file[0]){
        printf("loading word vectors...\n");
        ReadWordVector();
        if(word.vocab_size>0){
            printf("Successfully load %lld words with %lld dimentions from %s\n", word.vocab_size, word.layer_size, word.read_file);
            has_word = 1;
        }
    }
    if(0!=entity.read_file[0]){
        printf("loading entity vectors...\n");
        ReadEntityVector();
        if(entity.vocab_size>0)
            printf("Successfully load %lld entities with %lld dimentions from %s\n", entity.vocab_size, entity.layer_size, entity.read_file);
        printf("Successfully load %lld entity labels\n", label.vocab_size);
        has_entity = 1;
        
    }
    if(has_word) size = word.layer_size;
    if(has_entity) size = entity.layer_size;
    if(!has_word&&!has_entity){printf("error! no words and entities loaded!");exit(1);}
    
    while(1){
        GetItem(item);
        if (!strcmp(item, "EXIT")) break;
        if(has_word)
            word_index = SearchWordVocab(item);
        if(has_entity)
            entity_index = SearchEntityVocab(item);
        
        if(word_index==-1 && entity_index==-1){
            printf("Out of dictionary word or entity: %s!\n", item);
            continue;
        }
        if(word_index!=-1){
            for (a = 0; a < size; a++) word_vec[a] = 0;
            for (a = 0; a < size; a++) word_vec[a] += word.M[a + word_index * size];
            printf("\nWord: %s  Position in vocabulary: %d\n", item, word_index);
            FindNearest(N, word_vec);
        }
        if(entity_index!=-1){
            for (a = 0; a < size; a++) entity_vec[a] = 0;
            printf("\nEntity: %s  Position in vocabulary: %d\n", &entity.vocab[entity_index * MAX_STRING], entity_index);
            for (a = 0; a < size; a++) entity_vec[a] += entity.M[a + entity_index * size];
            FindNearest(N, entity_vec);
        }
        
    }
    
    return 0;
}
