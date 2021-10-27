import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that input is flatten images of size 28px by 28px
    '''
    
    def __init__(self, num_classes, optimizer, seed=1, train=True):

        # params
        self.num_classes = num_classes

        self.optimizer = optimizer
        self.use_in_train = train
        
        # create computation graph        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            if self.use_in_train:
                self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            else:
                self.features, self.labels, _, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            self.model_assign_op = []
            self.model_placeholder = []
            for variable in all_vars:
                self.model_placeholder.append(tf.placeholder(tf.float32))
            
            for variable, val in zip(all_vars, self.model_placeholder):
                self.model_assign_op.append(variable.assign(val))
            self.trainer_assign_model = tf.group(*self.model_assign_op)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        fc1 = tf.layers.dense(inputs=features, units=128, activation='relu')
        logits = tf.layers.dense(inputs=fc1, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)

        if self.use_in_train:
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        else:
            train_op = None
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        '''set model parameters'''
        if model_params is not None:
            feed_dict = {
               placeholder : value 
                  for placeholder, value in zip(self.model_placeholder, model_params)
            }
            with self.graph.as_default():
                self.sess.run(self.trainer_assign_model, feed_dict=feed_dict)

    def get_params(self):
        '''get model parameters'''
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        '''compute model gradient'''
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
        grads = process_grad(model_grads)

        return num_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}

        Return:
            loss and accuracy
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        '''clean up session'''
        self.sess.close()
