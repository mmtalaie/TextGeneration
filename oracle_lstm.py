import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from tqdm import tqdm
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Model:
    def __init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate, num_emb=10000):
        self.num_emb = num_emb

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, sequence_length, state_is_tuple=False
            )

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell() for _ in range(num_layers)], state_is_tuple=False
        )
        self.X = tf.placeholder(tf.int32, (None, None))
        self.Y = tf.placeholder(tf.int32, (None, None))
        embeddings = tf.Variable(
            tf.random_uniform([num_emb, size_layer], -1, 1)
        )
        encoder_embedded = tf.nn.embedding_lookup(embeddings, self.X)
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            self.rnn_cells,
            encoder_embedded,
            initial_state=self.hidden_layer,
            dtype=tf.float32,
        )
        self.logits = tf.layers.dense(self.outputs, dimension)
        logits_long = tf.reshape(self.logits, [-1, dimension])
        y_batch_long = tf.reshape(self.Y, [-1])
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_long, labels=y_batch_long
            )
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.correct_pred = tf.equal(
            tf.argmax(logits_long, 1), tf.cast(y_batch_long, tf.int64)
        )
        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))
        # supervised pretraining for generator
        self.g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=ms.seq_len,
            dynamic_size=False, infer_shape=True)

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.pretrain_loss = tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.X, [-1])), self.num_emb, 1.0, 0.0) *
            tf.log(tf.reshape(tf.reshape(self.g_predictions, [-1, self.num_emb]))) / (
                        self.sequence_length * self.batch_size)
        )

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.final_outputs = tf.nn.softmax(self.logits)


def get_vocab(file, lower=False):
    with open(file, 'r') as fopen:
        data = fopen.read()
    if lower:
        data = data.lower()
    data = data.split()
    vocab = list(set(data))
    return data, vocab


def embed_to_onehot(data, vocab):
    onehot = np.zeros((len(data)), dtype=np.int32)
    for i in range(len(data)):
        onehot[i] = vocab.index(data[i])
    return onehot


text, text_vocab = get_vocab('./dataset2/train.txt')
import model_settings as ms

learning_rate = 0.001
batch_size = 16
sequence_length = ms.seq_len
epoch = 3000
num_layers = 2
size_layer = 64
possible_batch_id = range(len(text) - sequence_length - 1)

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, len(text_vocab), sequence_length, learning_rate)
sess.run(tf.global_variables_initializer())


def train_random_sequence():
    LOST, ACCURACY = [], []
    pbar = tqdm(range(epoch), desc='epoch')
    for i in pbar:
        last_time = time.time()
        init_value = np.zeros((batch_size, num_layers * 2 * size_layer))
        batch_x = np.zeros((batch_size, sequence_length))
        batch_y = np.zeros((batch_size, sequence_length))
        batch_id = random.sample(possible_batch_id, batch_size)
        for n in range(sequence_length):
            id1 = embed_to_onehot([text[k + n] for k in batch_id], text_vocab)
            id2 = embed_to_onehot([text[k + n + 1] for k in batch_id], text_vocab)
            batch_x[:, n] = id1
            batch_y[:, n] = id2
        last_state, _, loss = sess.run([model.last_state, model.optimizer, model.loss],
                                       feed_dict={model.X: batch_x,
                                                  model.Y: batch_y,
                                                  model.hidden_layer: init_value})
        accuracy = sess.run(model.accuracy, feed_dict={model.X: batch_x,
                                                       model.Y: batch_y,
                                                       model.hidden_layer: init_value})
        ACCURACY.append(accuracy)
        LOST.append(loss)
        init_value = last_state
        pbar.set_postfix(cost=loss, accuracy=accuracy)
    return LOST, ACCURACY


LOST, ACCURACY = train_random_sequence()
