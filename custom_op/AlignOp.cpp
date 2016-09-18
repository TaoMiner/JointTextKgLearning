//
//  AlignOp.cpp
//  helloWorld
//
//  Created by 曹艺馨 on 16/8/23.
//  Copyright © 2016年 ethan. All rights reserved.
//

#include "AlignOp.hpp"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"

#include <fstream>
#include "simple_philox.h"
#include "philox_random.h"
#include <list>

REGISTER_OP("AlignModel")
    .Output("anchors_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("vocab_word: list(string)")
    .Attr("vocab_word_freq: list(int)")
    .Attr("vocab_entity: list(string)")
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("subsample: float = 1e-3")
    .Doc(R"doc(
Parses a text file and creates a batch of examples according to the given words dic, entity dic, and word freq.

vocab_word: A vector of words dic.
vocab_word_freq: Frequencies of words. Sorted in the non-ascending order.
vocab_entity: A vector of entity dic.
anchors_per_epoch: Number of anchors per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
subsample: Threshold for word occurrence. Words that appear with higher
    frequency will be randomly down-sampled. Set to 0 to disable.
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
    
    class AlignOp : public OpKernel {
    public:
        // 构造op时，读入全部文本，预加载部分样本，初始化变量。之后run时只输出compute方法结果
        explicit AlignOp(OpKernelConstruction* ctx)
        : OpKernel(ctx), rng_(&philox_){
            //read input tensor
            std::vector<string> words;
            std::vector<string> entities;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_word", &words));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_entity", &entities));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_word_freq", &words_freq_));

            // build word and entity dic
            const int N_words = words.size();
            for(auto i=0;i<N_words;i++){
                word_id_[words[i]] = i;
            }

            const int N_entities = entities.size();
            for(auto i=0;i<N_entities;i++){
                entity_id_[entities[i]] = i;
            }

            context_size_ = 0;
            const int N_words_freq = words_freq_.size();
            if(N_words_freq!=N_words)
                errors::InvalidArgument("The number of word vocab: ", N_words,
                                               " doesn't match its frequency size: ",
                                               N_words_freq);
            for(auto i=0;i<N_words_freq;i++){
                corpus_size_ += words_freq_[i];
            }

            string filename;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
            OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
            // 线程锁控制关键变量，如下声明的地方
            mutex_lock l(mu_);
            example_pos_ = sentence_num_-1;
            label_pos_ = corpus_[example_pos_].size()-1;
            label_limit_ = std::min<int32>(label_pos_+1, skip_+1);
            // 预加载kPrecalc个样本，按KSentenceSize长度将文本截断为句子。
            for (int i = 0; i < kPrecalc; ++i) {
                NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
            }
        }
        // 自定义op重写compute函数，ctx提供
        void Compute(OpKernelContext* ctx) override {
            Tensor anchors_per_epoch(DT_INT64, TensorShape({}));
            Tensor current_epoch(DT_INT32, TensorShape({}));
            Tensor total_anchor_processed(DT_INT64, TensorShape({}));
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
                anchors_per_epoch.scalar<int64>()() = sentence_num_;
                current_epoch.scalar<int32>()() = current_epoch_;
                total_anchor_processed.scalar<int64>()() = total_anchor_processed_;
            }
            ctx->set_output(0, anchors_per_epoch);
            ctx->set_output(1, current_epoch);
            ctx->set_output(2, total_anchor_processed);
            ctx->set_output(3, examples);
            ctx->set_output(4, labels);
        }
        
    private:
        struct Example {
            int32 input;
            int32 label;
        };
        
        int32 skip_ = 0;
        int32 batch_size_ = 0;
        int32 window_size_ = 5;
        float subsample_ = 1e-3;
        int32 context_size_ = 0;
        int32 corpus_size_ = 0;     //words size
        int32 sentence_num_ = 0;
        std::unordered_map<string, int32> word_id_;
        std::vector<int32> words_freq_;
        std::unordered_map<string, int32> entity_id_;
        std::vector<std::vector<int32>> corpus_;
        std::vector<Example> precalc_examples_;
        int precalc_index_ = 0;
        
        mutex mu_;
        random::PhiloxRandom philox_ GUARDED_BY(mu_);
        random::SimplePhilox rng_ GUARDED_BY(mu_);
        int32 current_epoch_ GUARDED_BY(mu_) = -1;
        int64 total_anchor_processed_ GUARDED_BY(mu_) = 0;
        int32 example_pos_ GUARDED_BY(mu_);
        int32 label_pos_ GUARDED_BY(mu_);
        int32 label_limit_ GUARDED_BY(mu_);
        
        // {example_pos_, label_pos_} is the cursor for the next example.
        // example_pos_ wraps around at the end of corpus_. For each
        // example, we randomly generate [label_pos_, label_limit) for
        // labels.
        // 构造样本，example纪录当前词，label为某一上下文，including subsample and skip window
        void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            while(true){
                if(label_pos_>=label_limit_-1){
                    label_pos_ = 0;
                    if(example_pos_>=sentence_num_-1){
                        example_pos_ = 0;
                        ++total_anchor_processed_;
                        skip_ = 1 + rng_.Uniform(window_size_);
                        ++current_epoch_;
                    }
                    else{
                        ++total_anchor_processed_;
                        ++example_pos_;
                        skip_ = 1 + rng_.Uniform(window_size_);
                    }
                    label_limit_ = std::min<int32>(corpus_[example_pos_].size(), skip_+1);
                }
                ++label_pos_;
                if (subsample_ > 0) {
                                int32 word_freq = words_freq_[corpus_[example_pos_][label_pos_]];
                                // See Eq. 5 in http://arxiv.org/abs/1310.4546
                                float keep_prob =
                                (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                                (subsample_ * corpus_size_) / word_freq;
                                auto tmp = rng_.RandFloat();

                                if ( tmp<= keep_prob) {
                                    break;
                                }
                                
                            }
                            else
                                break;
                            
            }
            *example = corpus_[example_pos_][0];
            *label = corpus_[example_pos_][label_pos_];
        }
        
        Status Init(Env* env, const string& filename) {
            //build corpus
            std::vector<string> line_words;
            corpus_.resize(5000000);
            std::ifstream fin(filename, std::ios::in);
            string line;
            int line_count = 0;
            int line_words_num = 0;
            context_size_ = 0;
            bool isEmpty = true;
            
            while(getline(fin, line)){
                line_words.clear();
                SplitString(line, line_words);
                line_words_num = line_words.size();
                if(line_words_num>2){
                    if(entity_id_.find(line_words[0])!=entity_id_.end())
                        corpus_[line_count].push_back(entity_id_[line_words[0]]);
                    else
                        continue;
                    isEmpty = true;
                    for(int i=1;i<line_words_num;i++){
                        if(word_id_.find(line_words[i])!=word_id_.end()){
                            corpus_[line_count].push_back(word_id_[line_words[i]]);
                            isEmpty = false;
                        }
                        else
                            continue;
                    }
                    if(isEmpty){
                        corpus_[line_count].clear();
                        continue;
                    }
                    context_size_ += (line_words_num-1);
 
                    line_count++;
                    if(line_count>=corpus_.size()){
                        corpus_.resize(corpus_.size()+100000);
                    }
                }
            }
            corpus_.resize(line_count);
            sentence_num_ = line_count;
            
            if (sentence_num_ < 5) {
                return errors::InvalidArgument("The anchor file ", filename,
                                               " contains too little data: ",
                                               sentence_num_, " anchors");
            }

            LOG(INFO) << "Data file: " << filename << " contains " << sentence_num_ << " anchors, "
            << context_size_ << " context words";
            precalc_examples_.resize(kPrecalc);
            skip_ = 1 + rng_.Uniform(window_size_);
            return Status::OK();
        }
    };
    
    REGISTER_KERNEL_BUILDER(Name("AlignModel").Device(DEVICE_CPU), AlignOp);
    
    
}  // end namespace tensorflow
