//
//  SGop.cpp
//  helloWorld
//
//  Created by 曹艺馨 on 16/8/23.
//  Copyright © 2016年 ethan. All rights reserved.
//

#include "KGop.hpp"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"

#include <fstream>

REGISTER_OP("KGskipgram")
    .Output("vocab_entity: string")
    .Output("vocab_freq: int32")
    .Output("entities_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_entities_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("min_count: int = 5")
    .Doc(R"doc(
Parses a text file and creates a batch of examples.

vocab_entity: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
entities_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_entities_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
)doc");


namespace tensorflow {
    
    // Number of examples to precalculate.
    const int kPrecalc = 3000;
    
    namespace {

        bool RemoveLeadingWhitespace(string& text) {
            std::string::size_type pos1, pos2;
            pos1 = text.find_first_not_of(" ");
            pos2 = text.find_last_not_of(" ");
            if(std::string::npos==pos1)
		      return false;
	        text = text.substr(pos1, pos2-pos1+1);
            return true;
        }

        void SplitString(const std::string& s, std::vector<std::string>& v)
        {
            std::string c = "\t\t";
            bool is_first = true;
            std::string::size_type pos1, pos2;
            pos2 = s.find(c);
            pos1 = 0;
            std::string tmp;
            while(std::string::npos != pos2)
            {
                tmp = s.substr(pos1, pos2-pos1);
                if(RemoveLeadingWhitespace(tmp))
                    v.push_back(tmp);
                
                pos1 = pos2 + c.size();
                if(is_first){
                    c = ";";
                    is_first = false;
                }
                pos2 = s.find(c, pos1);
            }
            if(pos1 != s.length()){
                tmp =s.substr(pos1);
                if(RemoveLeadingWhitespace(tmp))
                    v.push_back(tmp);
            }
        }
        
    }  // end namespace
    
    class KGop : public OpKernel {
    public:
        // 构造op时，读入全部文本，预加载部分样本，初始化变量。之后run时只输出compute方法结果
        explicit KGop(OpKernelConstruction* ctx)
        : OpKernel(ctx){
            string filename;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
            OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
            // 线程锁控制关键变量，如下声明的地方
            mutex_lock l(mu_);
            example_pos_ = sentence_num_-1;
            label_pos_ = corpus_[example_pos_].size()-1;
            // 预加载kPrecalc个样本，按KSentenceSize长度将文本截断为句子。
            for (int i = 0; i < kPrecalc; ++i) {
                NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
            }
        }
        // 自定义op重写compute函数，ctx提供
        void Compute(OpKernelContext* ctx) override {
            Tensor entities_per_epoch(DT_INT64, TensorShape({}));
            Tensor current_epoch(DT_INT32, TensorShape({}));
            Tensor total_entities_processed(DT_INT64, TensorShape({}));
            Tensor examples(DT_INT32, TensorShape({batch_size_}));
            auto Texamples = examples.flat<int32>();
            Tensor labels(DT_INT32, TensorShape({batch_size_}));
            auto Tlabels = labels.flat<int32>();
            {
                mutex_lock l(mu_);
                for (int i = 0; i < batch_size_; ++i) {
                    Texamples(i) = precalc_examples_[precalc_index_].input;
                    Tlabels(i) = precalc_examples_[precalc_index_].label;
                    precalc_index_++;
                    if (precalc_index_ >= kPrecalc) {
                        precalc_index_ = 0;
                        for (int j = 0; j < kPrecalc; ++j) {
                            NextExample(&precalc_examples_[j].input,
                                        &precalc_examples_[j].label);
                        }
                    }
                }
                entities_per_epoch.scalar<int64>()() = sentence_num_;
                current_epoch.scalar<int32>()() = current_epoch_;
                total_entities_processed.scalar<int64>()() = total_entities_processed_;
            }
            ctx->set_output(0, entity_);
            ctx->set_output(1, freq_);
            ctx->set_output(2, entities_per_epoch);
            ctx->set_output(3, current_epoch);
            ctx->set_output(4, total_entities_processed);
            ctx->set_output(5, examples);
            ctx->set_output(6, labels);
        }
        
    private:
        struct Example {
            int32 input;
            int32 label;
        };
        
        int32 batch_size_ = 0;
        int min_count_ = 5;
        int32 vocab_size_ = 0;
        Tensor entity_;
        Tensor freq_;
        int32 sentence_num_ = 0;
        std::vector<std::vector<int32>> corpus_;
        std::vector<Example> precalc_examples_;
        int precalc_index_ = 0;
        
        mutex mu_;
        int32 current_epoch_ GUARDED_BY(mu_) = -1;
        int64 total_entities_processed_ GUARDED_BY(mu_) = -1;
        int32 example_pos_ GUARDED_BY(mu_);
        int32 label_pos_ GUARDED_BY(mu_);
        
        // {example_pos_, label_pos_} is the cursor for the next example.
        // example_pos_ wraps around at the end of corpus_. For each
        // example, we randomly generate [label_pos_, label_limit) for
        // labels.
        // 构造样本，example纪录当前词，label为某一上下文，
        void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if(label_pos_>=corpus_[example_pos_].size()-1){
                label_pos_ = 0;
                if(example_pos_>=sentence_num_-1){
                    example_pos_ = 0;
                    ++total_entities_processed_;
                    ++current_epoch_;
                }
                else{
                    ++total_entities_processed_;
                    ++example_pos_;
                }
            }
            ++label_pos_;
            *example = corpus_[example_pos_][0];
            *label = corpus_[example_pos_][label_pos_];
        }
        
        Status Init(Env* env, const string& filename) {
            std::vector<std::vector<string>> raw_corpus(5200000);
            std::ifstream fin(filename, std::ios::in);
            string line;
            int line_count = 0;
            
            while(getline(fin, line)){
                line_count++;
                if(line_count>raw_corpus.size()){
                    raw_corpus.resize(raw_corpus.size()+100000);
                }
                SplitString(line, raw_corpus[line_count-1]);
            }
            
            sentence_num_ = line_count;
            raw_corpus.resize(sentence_num_);
            string w;
            std::unordered_map<string, int32> entity_freq;

            for(auto i = 0;i<raw_corpus.size();i++)
                for(auto j = 0;j<raw_corpus[i].size();j++){
                    ++(entity_freq[raw_corpus[i][j]]);
                }

            if (sentence_num_ < 50) {
                return errors::InvalidArgument("The knowledge file ", filename,
                                               " contains too little data: ",
                                               sentence_num_, " entitites");
            }
            typedef std::pair<string, int32> EntityFreq;
            std::vector<EntityFreq> ordered;
            for (const auto& p : entity_freq) {
                if (p.second >= min_count_) ordered.push_back(p);
            }
            LOG(INFO) << "Data file: " << filename << " contains " << sentence_num_ << " entities, " << entity_freq.size()
            << " unique entities, " << ordered.size()
            << " unique frequent entities.";
            entity_freq.clear();
            std::sort(ordered.begin(), ordered.end(),
                      [](const EntityFreq& x, const EntityFreq& y) {
                          return x.second > y.second;
                      });
            vocab_size_ = static_cast<int32>(ordered.size());
            Tensor entity(DT_STRING, TensorShape({vocab_size_}));
            Tensor freq(DT_INT32, TensorShape({vocab_size_}));

            std::unordered_map<string, int32> entity_id;
            int64 total_counted = 0;
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const auto& w = ordered[i].first;
                auto id = i;
                entity.flat<string>()(id) = w;
                auto entity_count = ordered[i].second;
                freq.flat<int32>()(id) = entity_count;
                total_counted += entity_count;
                entity_id[w] = id;
            }
            entity_ = entity;
            freq_ = freq;
            corpus_.resize(sentence_num_);
            sentence_num_ = 0;
            for(auto i=0, j=0;i<raw_corpus.size();i++){
                if(entity_id.find(raw_corpus[i][0])==entity_id.end())
                    continue;
                else
                    corpus_[j].push_back(entity_id[raw_corpus[i][0]]);
                for(auto k=1;k<raw_corpus[i].size();k++)
                    if(entity_id.find(raw_corpus[i][k])!=entity_id.end())
                        corpus_[j].push_back(entity_id[raw_corpus[i][k]]);
                if(corpus_[j].size()>1){
                    sentence_num_++;
                    j++;
                }
                else
                    corpus_[j].clear();
            }
            corpus_.resize(sentence_num_);
            precalc_examples_.resize(kPrecalc);
            return Status::OK();
        }
    };
    
    REGISTER_KERNEL_BUILDER(Name("KGskipgram").Device(DEVICE_CPU), KGop);
    
    
}  // end namespace tensorflow
