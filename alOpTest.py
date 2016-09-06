import tensorflow as tf
_align_model_module=tf.load_op_library('align_model.so')
align_model=_align_model_module.align_model
_kg_skipgram_module=tf.load_op_library('kg_skipgram.so')
kg_skipgram=_kg_skipgram_module.k_gskipgram
_text_skipgram_module=tf.load_op_library('skipgram.so')
skipgram=_text_skipgram_module.s_gjoint


with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        (entities, e_counts, entity_per_epoch, e_current_epoch, total_entity_processed, e_examples, e_labels) = kg_skipgram(filename="train_kg_sample",batch_size=5,min_count=0)
        (words, w_counts, words_per_epoch, w_current_epoch, total_words_processed,
     w_examples, w_labels) = skipgram(filename='train_text_sample',
                                           batch_size=5,
                                           window_size=5,
                                           min_count=0,
                                           subsample=1e-3)
        entity_dic,word_dic,word_freq= session.run([entities,words,w_counts])

        (anchors_per_epoch, a_current_epoch, total_anchors_processed,
     a_examples, a_labels) = align_model(filename='train_anchors_sample',
                                           batch_size=5,
                                           window_size=5,
                                           subsample=1e-3,vocab_word=word_dic.tolist(),vocab_word_freq=word_freq.tolist(),vocab_entity=entity_dic.tolist())

        print("#word: %d, #word fre: %d, #entity: %d" % (len(word_dic), len(word_freq), len(entity_dic)))
        epoch, has_processed, anchors_per_epoch_ = session.run([a_current_epoch, total_anchors_processed, anchors_per_epoch])
        print("initial epoch: %d, processed %d anchors, #anchors per epoch: %d" % (epoch,has_processed, anchors_per_epoch_))
        while has_processed<2000:
            examples, labels = session.run([a_examples, a_labels])
            for i in range(len(examples)):
                print("%s:%s" % (entity_dic[examples[i]], word_dic[labels[i]]))
            epoch, has_processed = session.run([a_current_epoch, total_anchors_processed])
            print("current epoch: %d, has processed: %d" % (epoch, has_processed))

