# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import codecs

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

# the kg op for build the entity example
_kg_skipgram_module=tf.load_op_library('kg_skipgram.so')
kg_skipgram=_kg_skipgram_module.k_gskipgram
# align model op for build the anchor example
_align_model_module=tf.load_op_library('align_model.so')
align_model=_align_model_module.align_model
# skipgram model op for build the text example
_text_model_module=tf.load_op_library('skipgram.so')
skipgram=_text_model_module.s_gjoint
neg_train=_text_model_module.joint_neg_train

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "text_data", None,
    "Text training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "kg_data", None,
    "Knowledge training data. E.g., ")
flags.DEFINE_string(
    "anchor_data", None,
    "Anchor text training data. E.g., ")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.text_data = FLAGS.text_data

    # The training knowledge file.
    self.kg_data = FLAGS.kg_data

    # The training anchor file.
    self.anchor_data = FLAGS.anchor_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    # including both word(0~vocab_word_size-1) and entity(0~vocab_entity_size-1)
    self._item2id = {}
    self._id2item = []
    self._v_in = []
    self._v_out = []

    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()
    self._read_analogies()
    self._model_names = ('text','knowledge','align')

  def _read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._item2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data for text skipgram. A text file.
    (words, w_counts, words_per_epoch, w_current_epoch, total_words_processed, w_examples, w_labels) = word2vec.skipgram(filename=opts.text_data, batch_size=opts.batch_size, window_size=opts.window_size, min_count=opts.min_count, subsample=opts.subsample)
    # the training data for entity skipgram
    (entities, e_counts, entities_per_epoch, e_current_epoch, total_entities_processed,
     e_examples, e_labels) = kg_skipgram(filename=opts.kg_data,
                                           batch_size=opts.batch_size,
                                           min_count=opts.min_count)

    (opts.vocab_words, vocab_word_counts,
     opts.words_per_epoch, opts.vocab_entities, vocab_entity_counts, opts.entities_per_epoch) = self._session.run([words, w_counts, words_per_epoch, entities, e_counts, entities_per_epoch])

    # the training data for align anchor skipgram
    (anchors_per_epoch, a_current_epoch, total_anchors_processed,
     a_examples, a_labels) = align_model(filename=opts.anchor_data,
                                           batch_size=opts.batch_size,
                                           window_size=opts.window_size,
                                           subsample=opts.subsample,vocab_word=opts.vocab_words.tolist(),vocab_word_freq=vocab_word_counts.tolist(),vocab_entity=opts.vocab_entities.tolist())


    opts.vocab_word_size = len(opts.vocab_words)
    opts.vocab_entity_size = len(opts.vocab_entities)
    opts.vocab_size = opts.vocab_word_size+opts.vocab_entity_size
    opts.anchors_per_epoch = self._session.run(anchors_per_epoch)

    # for neg sample, [vocab_word_counts, 0...0] and [0...0, vocab_entity_counts]
    opts.vocab_word_counts = tf.concat(0,[vocab_word_counts,tf.zeros([opts.vocab_entity_size])]).eval()
    opts.vocab_entity_counts = tf.concat(0,[tf.zeros([opts.vocab_word_size]), vocab_entity_counts]).eval()

    print("Text data file: ", opts.text_data)
    print("Word vocab size: ", opts.vocab_word_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    print("Entity data file: ", opts.kg_data)
    print("Entity vocab size: ", opts.vocab_entity_size)
    print("Entities per epoch: ", opts.entities_per_epoch)
    print("Anchor data file: ", opts.anchor_data)
    print("Anchors per epoch: ", opts.anchors_per_epoch)

    # for i< opts.vocab_word_size is a word, others is entity
    self._id2item = np.concatenate((opts.vocab_words, opts.vocab_entities), 0)
    for i, w in enumerate(self._id2item):
      self._item2id[w] = i

    # Declare all variables we need.
    # Input embedding including both words and entities: [vocab_size, emb_dim]
    # shard variable if larger than 2g, because saver's limit

    self._is_shard = False
    vocab_size_limit = 511000000/opts.emb_dim
    # 2048/4-1 mb
    if opts.vocab_size > vocab_size_limit:
      self._is_shard = True
      remain_size = opts.vocab_size
      for i in xrange(opts.vocab_size/vocab_size_limit):
        self._v_in.append(tf.Variable(tf.random_uniform([vocab_size_limit,opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),name="v_in"+i))
        self._v_out.append(tf.Variable(tf.zeros([vocab_size_limit,opts.emb_dim]),name="v_out"+i))
        remain_size -= vocab_size_limit
      if remain_size!=0:
        tmp_cap = len(self._v_in)
        self._v_in.append(tf.Variable(tf.random_uniform([remain_size,opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),name="v_in"+tmp_cap))
        self._v_out.append(tf.Variable(tf.zeros([remain_size,opts.emb_dim]),name="v_out"+tmp_cap))
    else:
      self._v_in.append(tf.Variable(tf.random_uniform([opts.vocab_size,opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),name="v_in"))
      self._v_out.append(tf.Variable(tf.zeros([opts.vocab_size,opts.emb_dim]),name="v_out"))

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      w_train = word2vec.neg_train(tf.concat(0,self._v_in) if self._is_shard else self._v_in[0],
                                 tf.concat(0,self._v_out) if self._is_shard else self._v_out[0],
                                 w_examples,
                                 w_labels,
                                 lr,
                                 vocab_count=opts.vocab_word_counts.tolist(),
                                 num_negative_samples=opts.num_samples)
      e_train = neg_train(tf.concat(0,self._v_in) if self._is_shard else self._v_in[0],
                                 tf.concat(0,self._v_out) if self._is_shard else self._v_out[0],
                                 e_examples+opts.vocab_word_size,
                                 e_labels+opts.vocab_word_size,
                                 lr,
                                 vocab_count=opts.vocab_entity_counts.tolist(),
                                 num_negative_samples=opts.num_samples)
      a_train = word2vec.neg_train(tf.concat(0,self._v_in) if self._is_shard else self._v_in[0], tf.concat(0,self._v_out) if self._is_shard else self._v_out[0], a_examples+opts.vocab_word_size, a_labels, lr, vocab_count=opts.vocab_word_counts.tolist(), num_negative_samples=opts.num_samples)


    self._lr = lr
    self._train_text = w_train
    self._train_kg = e_train
    self._train_align = a_train
    self.step = global_step
    self._epoch_text = w_current_epoch
    self._epoch_kg = e_current_epoch
    self._epoch_align = a_current_epoch
    self._words = total_words_processed
    self._entities = total_entities_processed
    self._anchors = total_anchors_processed

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with codecs.open(os.path.join(opts.save_path, "vocab_words.txt"), "w", encoding='UTF-8') as f:
      for i in xrange(opts.vocab_word_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]), opts.vocab_word_counts[i]))

    with codecs.open(os.path.join(opts.save_path, "vocab_entities.txt"), "w", encoding='UTF-8') as f:
      for i in xrange(opts.vocab_entity_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_entities[i]),
                             opts.vocab_entity_counts[opts.vocab_word_size+i]))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(tf.concat(0,self._v_in) if self._is_shard else self._v_in[0], 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_word_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()

    # model_index: 1--text model; 2--kg model; 3--align model
  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])

    while True:
      _, tmp_epoch = self._session.run([self._train, self._epoch])
      if tmp_epoch != initial_epoch:
        break

  def train(self):
    opts = self._options
    model_index = 0
    while model_index<3:
      """initial the model parameter"""
      if model_index == 0:
        self._epoch = self._epoch_text
        self._items = self._words
        self._train = self._train_text
      elif model_index == 1:
        self._epoch = self._epoch_kg
        self._items = self._entities
        self._train = self._train_kg
      elif model_index == 2:
        self._epoch = self._epoch_align
        self._items = self._anchors
        self._train = self._train_align
      else:
        print("there is no model: %d" % model_index)
        break

      """Train the model."""
      initial_epoch, initial_items = self._session.run([self._epoch, self._items])

      workers = []
      for _ in xrange(opts.concurrent_steps):
        t = threading.Thread(target=self._train_thread_body)
        t.start()
        workers.append(t)

      last_items, last_time = initial_items, time.time()
      while True:
        time.sleep(5)  # Reports our progress once a while.
        (tmp_epoch, step, tmp_items,
         lr) = self._session.run([self._epoch, self.step, self._items, self._lr])
        now = time.time()
        last_items, last_time, rate = tmp_items, now, (tmp_items - last_items) / (
            now - last_time)

        print("%s model, Epoch %4d Step %8d: lr = %5.3f items/sec = %8.0f\r" % (self._model_names[model_index], tmp_epoch, step, lr, rate), end="")
        sys.stdout.flush()
        if tmp_epoch != initial_epoch:
          break

      for t in workers:
        t.join()

      model_index += 1
      self.eval()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    total = self._analogy_questions.shape[0]
    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._item2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2item[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        return c
    return "unknown"

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._item2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2item[neighbor], distance))


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  if not FLAGS.text_data or not FLAGS.kg_data or not FLAGS.anchor_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--text_data --kg_data --anchor_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
    # Perform a final save.
    model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.step)
    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy(b'france', b'paris', b'russia')
      # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
      _start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
