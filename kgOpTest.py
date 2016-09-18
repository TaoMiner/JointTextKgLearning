import tensorflow as tf
_kg_skipgram_module=tf.load_op_library('kg_skipgram.so')
kg_skipgram=_kg_skipgram_module.k_gskipgram

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        (entities, counts, entity_per_epoch, current_epoch, total_entity_processed, examples, labels) = kg_skipgram(filename="train_kg_sample",batch_size=5,min_count=0)

        entity_dic, entity_fre, entity_per = session.run([entities, counts, entity_per_epoch])
        print(entity_dic)
        print("#entity: %d, #entity fre: %d, #entity pre epoch: %d" % (len(entity_dic), len(entity_fre), entity_per))
        epoch, has_processed = session.run([current_epoch, total_entity_processed])
        print("initial epoch: %d, processed %d entities." % (epoch,has_processed))
        while has_processed<10:
            e_examples, e_labels = session.run([examples, labels])
            for i in range(len(e_examples)):
                print("%s:%s" % (entity_dic[e_examples[i]], entity_dic[e_labels[i]]))
            epoch, has_processed = session.run([current_epoch, total_entity_processed])
            print("current epoch: %d, has processed: %d" % (epoch, has_processed))
