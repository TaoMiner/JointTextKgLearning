//
//  SGop.cpp
//  helloWorld
//
//  Created by 曹艺馨 on 16/8/23.
//  Copyright © 2016年 ethan. All rights reserved.
//

#include "SGop.hpp"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"

#include "simple_philox.h"
#include "philox_random.h"
#include "distribution_sampler.h"

namespace tensorflow {

    REGISTER_OP("SGjoint")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 5")
    .Attr("subsample: float = 1e-3")
    .Doc(R"doc(
Parses a text file and creates a batch of examples.

vocab_word: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the left and right of the target.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
subsample: Threshold for word occurrence. Words that appear with higher
    frequency will be randomly down-sampled. Set to 0 to disable.
)doc");

REGISTER_OP("JointNegTrain")
    .Input("w_in: Ref(float)")
    .Input("w_out: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .Doc(R"doc(
Training via negative sampling.

w_in: input word embedding.
w_out: output word embedding.
examples: A vector of word ids.
labels: A vector of word ids.
vocab_count: Count of words in the vocabulary.
num_negative_samples: Number of negative samples per example.
)doc");
    
    // Number of examples to precalculate.
    const int kPrecalc = 3000;
    // Number of words to read into a sentence before processing.
    const int kSentenceSize = 1000;
    
    namespace {
        
        template <class Collection>
        const typename Collection::value_type::second_type& FindWithDefault(
                                                                            const Collection& collection,
                                                                            const typename Collection::value_type::first_type& key,
                                                                            const typename Collection::value_type::second_type& value) {
            typename Collection::const_iterator it = collection.find(key);
            if (it == collection.end()) {
                return value;
            }
            return it->second;
        }
        
        bool ScanWord(StringPiece* input, string* word) {
            str_util::RemoveLeadingWhitespace(input);
            StringPiece tmp;
            if (str_util::ConsumeNonWhitespace(input, &tmp)) {
                word->assign(tmp.data(), tmp.size());
                return true;
            } else {
                return false;
            }
        }
        
    }  // end namespace
    
    class GuardedPhiloxRandom {
    public:
        // Must call Init to finish initialization
        GuardedPhiloxRandom() : initialized_(false) {}
        
        // Initialize the generator from attributes "seed" and "seed2".
        // If both seeds are unspecified, use random seeds.
        // Must be called exactly once.
        Status Init(OpKernelConstruction* context);
        
        // Initialize with given seeds.
        void Init(int64 seed, int64 seed2);
        
        // Reserve a certain number of 128-bit samples.
        // This function is thread safe.  The returned generator is valid for the
        // given number of samples, and can be used without a lock.
        random::PhiloxRandom ReserveSamples128(int64 samples);
        
        // Reserve a certain number of 32-bit samples.
        random::PhiloxRandom ReserveSamples32(int64 samples) {
            return ReserveSamples128((samples + 3) / 4);
        }
        
        // Reserve enough random samples in the generator for the given output count.
        random::PhiloxRandom ReserveRandomOutputs(int64 output_count,
                                                  int multiplier) {
            int64 conservative_sample_count = output_count * multiplier;
            return ReserveSamples128(conservative_sample_count);
        }
        
    private:
        mutex mu_;
        random::PhiloxRandom generator_ GUARDED_BY(mu_);
        bool initialized_;
        
        TF_DISALLOW_COPY_AND_ASSIGN(GuardedPhiloxRandom);
    };
    
    class SGJointOp : public OpKernel {
    public:
        // 构造op时，读入全部文本，预加载部分样本，初始化变量。之后run时只输出compute方法结果
        explicit SGJointOp(OpKernelConstruction* ctx)
        : OpKernel(ctx), rng_(&philox_) {
            string filename;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
            OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
            // 线程锁控制关键变量，如下声明的地方
            mutex_lock l(mu_);
            example_pos_ = corpus_size_;
            label_pos_ = corpus_size_;
            label_limit_ = corpus_size_;
            sentence_index_ = kSentenceSize;
            // 预加载kPrecalc个样本，按KSentenceSize长度将文本截断为句子。
            for (int i = 0; i < kPrecalc; ++i) {
                NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
            }
        }
        // 自定义op重写compute函数，ctx提供
        void Compute(OpKernelContext* ctx) override {
            Tensor words_per_epoch(DT_INT64, TensorShape({}));
            Tensor current_epoch(DT_INT32, TensorShape({}));
            Tensor total_words_processed(DT_INT64, TensorShape({}));
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
                words_per_epoch.scalar<int64>()() = corpus_size_;
                current_epoch.scalar<int32>()() = current_epoch_;
                total_words_processed.scalar<int64>()() = total_words_processed_;
            }
            ctx->set_output(0, word_);
            ctx->set_output(1, freq_);
            ctx->set_output(2, words_per_epoch);
            ctx->set_output(3, current_epoch);
            ctx->set_output(4, total_words_processed);
            ctx->set_output(5, examples);
            ctx->set_output(6, labels);
        }
        
    private:
        struct Example {
            int32 input;
            int32 label;
        };
        
        int32 batch_size_ = 0;
        int32 window_size_ = 5;
        float subsample_ = 1e-3;
        int min_count_ = 5;
        int32 vocab_size_ = 0;
        Tensor word_;
        Tensor freq_;
        int32 corpus_size_ = 0;
        std::vector<int32> corpus_;
        std::vector<Example> precalc_examples_;
        int precalc_index_ = 0;
        std::vector<int32> sentence_;
        int sentence_index_ = 0;
        
        mutex mu_;
        random::PhiloxRandom philox_ GUARDED_BY(mu_);
        random::SimplePhilox rng_ GUARDED_BY(mu_);
        int32 current_epoch_ GUARDED_BY(mu_) = -1;
        int64 total_words_processed_ GUARDED_BY(mu_) = 0;
        int32 example_pos_ GUARDED_BY(mu_);
        int32 label_pos_ GUARDED_BY(mu_);
        int32 label_limit_ GUARDED_BY(mu_);
        
        // {example_pos_, label_pos_} is the cursor for the next example.
        // example_pos_ wraps around at the end of corpus_. For each
        // example, we randomly generate [label_pos_, label_limit) for
        // labels.
        // 构造样本，example纪录当前词，label为某一上下文，
        void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            while (true) {
                if (label_pos_ >= label_limit_) {
                    ++total_words_processed_;
                    ++sentence_index_;
                    if (sentence_index_ >= kSentenceSize) {
                        sentence_index_ = 0;
                        for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
                            if (example_pos_ >= corpus_size_) {
                                ++current_epoch_;
                                example_pos_ = 0;
                            }
                            if (subsample_ > 0) {
                                int32 word_freq = freq_.flat<int32>()(corpus_[example_pos_]);
                                // See Eq. 5 in http://arxiv.org/abs/1310.4546
                                float keep_prob =
                                (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                                (subsample_ * corpus_size_) / word_freq;
                                if (rng_.RandFloat() > keep_prob) {
                                    i--;
                                    continue;
                                }
                            }
                            sentence_[i] = corpus_[example_pos_];
                        }
                    }
                    const int32 skip = 1 + rng_.Uniform(window_size_);
                    label_pos_ = std::max<int32>(0, sentence_index_ - skip);
                    label_limit_ =
                    std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
                }
                if (sentence_index_ != label_pos_) {
                    break;
                }
                ++label_pos_;
            }
            *example = sentence_[sentence_index_];
            *label = sentence_[label_pos_++];
        }
        
        Status Init(Env* env, const string& filename) {
            string data;
            TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
            StringPiece input = data;
            string w;
            corpus_size_ = 0;
            std::unordered_map<string, int32> word_freq;
            while (ScanWord(&input, &w)) {
                ++(word_freq[w]);
                ++corpus_size_;
            }
            if (corpus_size_ < window_size_ * 10) {
                return errors::InvalidArgument("The text file ", filename,
                                               " contains too little data: ",
                                               corpus_size_, " words");
            }
            typedef std::pair<string, int32> WordFreq;
            std::vector<WordFreq> ordered;
            for (const auto& p : word_freq) {
                if (p.second >= min_count_) ordered.push_back(p);
            }
            LOG(INFO) << "Data file: " << filename << " contains " << data.size()
            << " bytes, " << corpus_size_ << " words, " << word_freq.size()
            << " unique words, " << ordered.size()
            << " unique frequent words.";
            word_freq.clear();
            std::sort(ordered.begin(), ordered.end(),
                      [](const WordFreq& x, const WordFreq& y) {
                          return x.second > y.second;
                      });
            vocab_size_ = static_cast<int32>(1 + ordered.size());
            Tensor word(DT_STRING, TensorShape({vocab_size_}));
            Tensor freq(DT_INT32, TensorShape({vocab_size_}));
            word.flat<string>()(0) = "UNK";
            static const int32 kUnkId = 0;
            std::unordered_map<string, int32> word_id;
            int64 total_counted = 0;
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const auto& w = ordered[i].first;
                auto id = i + 1;
                word.flat<string>()(id) = w;
                auto word_count = ordered[i].second;
                freq.flat<int32>()(id) = word_count;
                total_counted += word_count;
                word_id[w] = id;
            }
            freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
            word_ = word;
            freq_ = freq;
            corpus_.reserve(corpus_size_);
            input = data;
            while (ScanWord(&input, &w)) {
                corpus_.push_back(FindWithDefault(word_id, w, kUnkId));
            }
            precalc_examples_.resize(kPrecalc);
            sentence_.resize(kSentenceSize);
            return Status::OK();
        }
    };
    
    REGISTER_KERNEL_BUILDER(Name("SGjoint").Device(DEVICE_CPU), SGJointOp);
    
    
    class JointNegTrainOp : public OpKernel {
    public:
        explicit JointNegTrainOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            base_.Init(0, 0);
            
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));
            
            std::vector<int32> vocab_count;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));
            
            std::vector<float> vocab_weights;
            vocab_weights.reserve(vocab_count.size());
            for (const auto& f : vocab_count) {
                float r = std::pow(static_cast<float>(f), 0.75f);
                vocab_weights.push_back(r);
            }
            sampler_ = new random::DistributionSampler(vocab_weights);
        }
        
        ~JointNegTrainOp() { delete sampler_; }
        
        void Compute(OpKernelContext* ctx) override {
            Tensor w_in = ctx->mutable_input(0, false);
            OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                        errors::InvalidArgument("Must be a matrix"));
            Tensor w_out = ctx->mutable_input(1, false);
            //OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
            //            errors::InvalidArgument("w_in.shape == w_out.shape"));
            const Tensor& examples = ctx->input(2);
            OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                        errors::InvalidArgument("Must be a vector"));
            const Tensor& labels = ctx->input(3);
            OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                        errors::InvalidArgument("examples.shape == labels.shape"));
            const Tensor& learning_rate = ctx->input(4);
            OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                        errors::InvalidArgument("Must be a scalar"));
            
            auto Tw_in = w_in.matrix<float>();
            auto Tw_out = w_out.matrix<float>();
            auto Texamples = examples.flat<int32>();
            auto Tlabels = labels.flat<int32>();
            auto lr = learning_rate.scalar<float>()();
            const int64 vocab_size = w_in.dim_size(0);
            const int64 dims = w_in.dim_size(1);
            const int64 batch_size = examples.dim_size(0);
            //OP_REQUIRES(ctx, vocab_size == sampler_->num(),
            //            errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
            //                                    " vs. ", sampler_->num()));
            
            // Gradient accumulator for v_in.
            Tensor buf(DT_FLOAT, TensorShape({dims}));
            auto Tbuf = buf.flat<float>();
            
            // Scalar buffer to hold sigmoid(+/- dot).
            Tensor g_buf(DT_FLOAT, TensorShape({}));
            auto g = g_buf.scalar<float>();
            
            // The following loop needs 2 random 32-bit values per negative
            // sample.  We reserve 8 values per sample just in case the
            // underlying implementation changes.
            auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
            random::SimplePhilox srnd(&rnd);
            
            for (int64 i = 0; i < batch_size; ++i) {
                const int32 example = Texamples(i);
                // DCHECK(0 <= example && example < vocab_size) << example;
                const int32 label = Tlabels(i);
                DCHECK(0 <= label && label < vocab_size) << label;
                auto v_in = Tw_in.chip<0>(example);
                
                // Positive: example predicts label.
                //   forward: x = v_in' * v_out
                //            l = log(sigmoid(x))
                //   backward: dl/dx = g = sigmoid(-x)
                //             dl/d(v_in) = g * v_out'
                //             dl/d(v_out) = v_in' * g
                {
                    auto v_out = Tw_out.chip<0>(label);
                    auto dot = (v_in * v_out).sum();
                    g = (dot.exp() + 1.f).inverse();
                    Tbuf = v_out * (g() * lr);
                    v_out += v_in * (g() * lr);
                }
                
                // Negative samples:
                //   forward: x = v_in' * v_sample
                //            l = log(sigmoid(-x))
                //   backward: dl/dx = g = -sigmoid(x)
                //             dl/d(v_in) = g * v_out'
                //             dl/d(v_out) = v_in' * g
                for (int j = 0; j < num_samples_; ++j) {
                    const int sample = sampler_->Sample(&srnd);
                    if (sample == label) continue;  // Skip.
                    auto v_sample = Tw_out.chip<0>(sample);
                    auto dot = (v_in * v_sample).sum();
                    g = -((-dot).exp() + 1.f).inverse();
                    Tbuf += v_sample * (g() * lr);
                    v_sample += v_in * (g() * lr);
                }
                
                // Applies the gradient on v_in.
                v_in += Tbuf;
            }
        }
        
    private:
        int32 num_samples_ = 0;
        random::DistributionSampler* sampler_ = nullptr;
        GuardedPhiloxRandom base_;
    };
    
    REGISTER_KERNEL_BUILDER(Name("JointNegTrain").Device(DEVICE_CPU), JointNegTrainOp);
    
}  // end namespace tensorflow
