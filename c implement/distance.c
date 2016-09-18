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

#define MAX_STRING 100

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries
const char split_pattern = '\t';

struct vocab_item{
    char read_file[MAX_STRING];
    char *vocab;
    long long vocab_size, layer_size;
    float *M;
    char label[MAX_STRING];
}word, entity;

long long size;

void FindNearest(int top_n, float *vec){
    char *bestw[top_n];
    float dist, bestd[top_n];
    long long a, c, d;
    float len;
    for (a = 0; a < top_n; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    for (a = 0; a < top_n; a++) bestd[a] = 0;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    
    //normalization
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    //compute nearest words and entities

    printf("\n                                              %s       Cosine distance\n------------------------------------------------------------------------\n", word.label);
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
                strcpy(bestw[a], &word.vocab[c * max_w]);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    
    printf("\n                                              %s       Cosine distance\n------------------------------------------------------------------------\n", entity.label);
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
                strcpy(bestw[a], &entity.vocab[c * max_w]);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
}

int SearchVocab(char *item, struct vocab_item *v_item){
    int b;
    for (b = 0; b < v_item->vocab_size; b++) if (!strcmp(&v_item->vocab[b * max_w], item)) break;
    if (b == v_item->vocab_size) b = -1;
    return b;
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

void ReadVector(struct vocab_item *item){
    FILE *f;
    long long a, b;
    float len;
    f = fopen(item->read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &item->vocab_size);
    fscanf(f, "%lld", &item->layer_size);
    item->vocab = (char *)malloc((long long)item->vocab_size * max_w * sizeof(char));
    item->M = (float *)malloc((long long)item->vocab_size * (long long)item->layer_size * sizeof(float));
    if (item->M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)item->vocab_size * item->layer_size * sizeof(float) / 1048576, item->vocab_size, item->layer_size);
        return;
    }
    for (b = 0; b < item->vocab_size; b++) {
        a = 0;
        while (1) {
            item->vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (item->vocab[b * max_w + a] == split_pattern)) break;
            if ((a < max_w) && (item->vocab[b * max_w + a] != '\n')) a++;
        }
        item->vocab[b * max_w + a] = 0;
        for (a = 0; a < item->layer_size; a++) fread(&item->M[a + b * item->layer_size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < item->layer_size; a++) len += item->M[a + b * item->layer_size] * item->M[a + b * item->layer_size];
        len = sqrt(len);
        for (a = 0; a < item->layer_size; a++) item->M[a + b * item->layer_size] /= len;
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
        printf("\nExamples:\n");
        printf("./distance -read_word_vector ./vec_word read_entity_vector ./vec_entity\n\n");
        return 0;
    }
    word.read_file[0] = 0;
    entity.read_file[0] = 0;
    word.vocab_size = 0;
    entity.vocab_size = 0;
    word.layer_size = 0;
    entity.layer_size = 0;
    if ((i = ArgPos((char *)"-read_word_vector", argc, argv)) > 0) strcpy(word.read_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_entity_vector", argc, argv)) > 0) strcpy(entity.read_file, argv[i + 1]);
    
    if(0!=word.read_file[0]){
        printf("loading word vectors...\n");
        ReadVector(&word);
        if(word.vocab_size>0)
            printf("Successfully load %lld words with %lld dimentions from %s\n", word.vocab_size, word.layer_size, word.read_file);
        has_word = 1;
        sprintf(word.label, "Word");
    }
    if(0!=entity.read_file[0]){
        printf("loading entity vectors...\n");
        ReadVector(&entity);
        if(entity.vocab_size>0)
            printf("Successfully load %lld entities with %lld dimentions from %s\n", entity.vocab_size, entity.layer_size, entity.read_file);
        has_entity = 1;
        sprintf(entity.label, "Entity");
    }
    if(has_word&&has_entity){
        if(word.layer_size!=entity.layer_size){
            printf("word dimention and entity dimention don't match!\n");
            return 0;
        }
        size = word.layer_size;
        //search in the word and entity vocab
        
    }
    if(has_word&&!has_entity){
        size = word.layer_size;
        //search in the word vocab
    }
    if(!has_word&&has_entity){
        size = entity.layer_size;
        //search in the entity vocab
    }
    while(1){
        GetItem(item);
        if (!strcmp(item, "EXIT")) break;
        if(has_word)
            word_index = SearchVocab(item, &word);
        if(has_entity)
            entity_index = SearchVocab(item, &entity);
        
        if(word_index==-1 && entity_index==-1){
            printf("Out of dictionary word or entity: %s!\n", item);
            break;
        }
        if(word_index!=-1){
            for (a = 0; a < size; a++) word_vec[a] = 0;
            for (a = 0; a < size; a++) word_vec[a] += word.M[a + word_index * size];
            printf("\nWord: %s  Position in vocabulary: %d\n", item, word_index);
            FindNearest(N, word_vec);
        }
        if(entity_index!=-1){
            for (a = 0; a < size; a++) entity_vec[a] = 0;
            printf("\nEntity: %s  Position in vocabulary: %d\n", item, entity_index);
            for (a = 0; a < size; a++) entity_vec[a] += entity.M[a + entity_index * size];
            FindNearest(N, entity_vec);
        }
        
    }
    
    return 0;
}
