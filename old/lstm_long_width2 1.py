# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '3,5'
# 路径设置
TRAIN_PERCENTAGE = 0.4
DATA_PATH = r'/Users/heningzhang/Desktop/dataset_UCL' # 数据路径
TRA_NUM = np.array([300]) # 各用户的轨迹数目
MODEL_SAVE_PATH =  "/Users/heningzhang/Desktop/dataset_UCL/model3/"
MODEL_LOAD_PATH = "/Users/heningzhang/Desktop/dataset_UCL/model3/"
# 数据打乱标识
FILE_SHUFFLE = False
BATCH_SHUFFLE = True
# 数据结构
DIMENSION_IN = 2
DIMENSION_OUT = 2
HIS_WIN = 100 # 历史窗长
PRE_WIN = 1 # 预测窗长
outSizeOfCEll = DIMENSION_OUT * PRE_WIN
TIME_STEPS = HIS_WIN + PRE_WIN # 数据的截断长度
# 结构参数
HIDDEN_SIZE = 200
NUM_LAYERS = 1
# 训练参数
NUM_EPOCHS = 20
LEARNING_RATE = 0.00003#0.006#0.00003
MAX_GRAD_NORM = 5
KEEP_PROB = 0.5
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 20
# 多次使用的变量
Predict_Index = -1 * PRE_WIN


def train_divide(dataPath, trainPercentage, traNum, shuffle = False):
    train_file_list = []
    test_file_list = []
    #
    for root, dirs, files in os.walk(dataPath):
        for dir in dirs:#进入第一层目录dev0
            for root2, dirs2, files2 in os.walk(os.path.join(dataPath, dir)):
                for dir2 in dirs2:#进入第二层目录
                    new_dataPath = os.path.join(dataPath, dir,dir2)
                    train_num = len(os.listdir(new_dataPath)) * trainPercentage
                    j = 0
                    for root3, dirs3, files3 in os.walk(new_dataPath):
                        for name in files3:
                            second_path = os.path.join(new_dataPath, name)
                            if j < train_num:
                                train_file_list.append(second_path)
                                j = j + 1 
                            else:
                                test_file_list.append(second_path)
    return train_file_list, test_file_list

def delete_nan(train_data):
    for i in range(len(train_data)):
        single = train_data[i]
        for j in range(len(single)):
            single2 = single[j]
            for k in range(len(single2)):
                if np.isnan(single2[k]) or np.isinf(single2[k]):
                    train_data[i][j][k] = 0
    return train_data

def get_train_data(train_file_list):
    train_data = []
    train_original = np.zeros(shape=(1, 2)) 
    print('Getting Training Data')
    k = 0
    for i in train_file_list:
        #print(i)
        try:
            res = pd.read_csv(i,header=None) #
            res.columns = ['azimuth', 'elevation', 'timestamp']
            #print(res)
            tra_data = np.zeros(shape=(len(res), 2))
            #print(len(res))
            for row in range(len(res)):
                #print(res.iloc[[row]]['azimuth'])
                tra_data[row][0] = res.iloc[[row]]['azimuth']
                #print(res[row]['azimuth'])
                tra_data[row][1] = res.iloc[[row]]['elevation']
                # tra_data[row][0] = res.values[row][4]*res.values[row][15]
                # tra_data[row][1] = res.values[row][5]*res.values[row][16]
            train_original = np.concatenate([train_original, tra_data])
            batch_num_temp = tra_data.shape[0] + 1 - TIME_STEPS
            for j in range(batch_num_temp):
                batch_data_temp = tra_data[j: j + TIME_STEPS]
                train_data.append(batch_data_temp)
            k += 1
            if k % 50 == 0:
                print(k)
        except:
            pass
    train_original = np.delete(train_original, 0, 0)
    train_data = np.array(train_data) # ******训练集太大时，分批读入********
    train_data = delete_nan(train_data)
    scaler = preprocessing.StandardScaler().fit(train_original)
    # 训练数据预处理
    train_data = (train_data - scaler.mean_) / scaler.scale_
    return train_data, scaler


def get_eval_data(eval_file_list, scaler):
    print('Getting Eval Data')
    eval_data = []
    k = 0
    for i in eval_file_list[0:10]:
        #tra_data = np.loadtxt(i, delimiter=',')
        try:
            res = pd.read_csv(i,header=None)
            res.columns = ['azimuth', 'elevation', 'timestamp']
            tra_data = np.zeros(shape=(len(res), 2))
            for row in range(len(res)):
                tra_data[row][0] = res.iloc[[row]]['azimuth']
                tra_data[row][1] = res.iloc[[row]]['elevation']
            tra_size = tra_data.shape[0]
            if (tra_size - TIME_STEPS) % PRE_WIN == 0:
                batch_end_index = tra_size - TIME_STEPS
            else:
                batch_end_index = tra_size - HIS_WIN - 1
                batch_end_index = batch_end_index - batch_end_index % PRE_WIN
            for start_index in range(0, batch_end_index, PRE_WIN):
                end_index = min(start_index +  TIME_STEPS, tra_size)
                batch_data_temp = tra_data[start_index: end_index]
                eval_data.append(batch_data_temp)
            if k % 50 == 0:
                print(k)
        except:
            pass
        # 测试数据预处理
    eval_data = np.array(eval_data)
    eval_data = delete_nan(eval_data)
    eval_data = (eval_data - scaler.mean_) / scaler.scale_
    return eval_data


def batch_iter(data, batch_size, num_epochs, shuffle = True):
    data_size = len(data) # data是np.array类型
    train_batch_num = data_size // batch_size
    for epoch in range(num_epochs):
        if shuffle:
            # 数据打乱
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
        for batch_index in range(train_batch_num):
            start_index_b = batch_index * batch_size
            end_index_b = (batch_index + 1) * batch_size
            yield data[start_index_b:end_index_b, :Predict_Index], data[start_index_b:end_index_b, Predict_Index:].reshape([batch_size,-1])


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, is_training):
        #　self.reuse_index = reuse_index
        self.is_training = is_training
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        # self.keep_prob = keep_prob
        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()
        self.compute_cost()
        if not is_training: return
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def add_input_layer(self,):
        with tf.variable_scope('input', reuse=not self.is_training):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
            Ws_in = self._weight_variable([self.input_size, self.cell_size])  # Ws (in_size, cell_size)
            bs_in = self._bias_variable([self.cell_size,])   # bs (cell_size, )
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in        # l_in_y = (batch * n_steps, cell_size)
            self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
            # reshape l_in_y ==> (batch, n_steps, cell_size)

    def add_cell_(self):
        basic_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        if self.is_training:
            basic_cell = tf.contrib.rnn.DropoutWrapper(cell=basic_cell, input_keep_prob=1.0, output_keep_prob=KEEP_PROB)
        return basic_cell
        
    def add_cell(self):
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        # lstm_cell.append(tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True))
        lstm_cell = tf.contrib.rnn.MultiRNNCell([self.add_cell_()] * NUM_LAYERS)
        self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
        self.cell_outputs = self.cell_outputs[:,-1]

    def add_output_layer(self):
        with tf.variable_scope('output',reuse=not self.is_training):
            l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')  # shape = (batch * steps, cell_size)
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # shape = (batch * steps, output_size)

    def compute_cost(self):
        """
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error, name='losses')
        self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size, name='average_cost')
        """
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.ys, self.pred)))
        # self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.ys, logits=self.pred)
    """
    def ms_error(self, labels, logits):
        #print np.shape(labels)
        #print np.shape(logits)
        a = tf.square(tf.subtract(labels, logits))
        print np.shape(a)
        return a
    """
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

def run_eval(sess, model, eval_data, eval_batch_num):
    eval_iter = batch_iter(eval_data, EVAL_BATCH_SIZE, 1, False)
    total_costs = 0.0
    iters = 0
    for step in range(eval_batch_num):
        X_in, Y_in = next(eval_iter)
        feed_dict = {
            model.xs: X_in,
            model.ys: Y_in
            }
        _, cost = sess.run([tf.no_op(), model.cost],
                           feed_dict=feed_dict)
        total_costs += cost
        iters += 1
        mse = total_costs / iters
        print("Root Mean Square Error is %f" % mse)


def main():
    # 数据预处理
    train_file_list, eval_file_list = train_divide(DATA_PATH, TRAIN_PERCENTAGE, TRA_NUM)
    train_data, scaler = get_train_data(train_file_list)
    eval_data = get_eval_data(eval_file_list, scaler)
    print(len(eval_data))
    train_iter = batch_iter(train_data, TRAIN_BATCH_SIZE, NUM_EPOCHS, shuffle=BATCH_SHUFFLE)

    with tf.variable_scope("long_model", reuse=None):
        train_model = LSTMRNN(HIS_WIN, DIMENSION_IN, DIMENSION_OUT ,outSizeOfCEll, TRAIN_BATCH_SIZE, True)
    with tf.variable_scope("long_model", reuse=True):
        eval_model = LSTMRNN(HIS_WIN, DIMENSION_IN, DIMENSION_OUT , outSizeOfCEll,EVAL_BATCH_SIZE, False)
    train_batch_num = len(train_data) // TRAIN_BATCH_SIZE
    eval_batch_num = len(eval_data) // EVAL_BATCH_SIZE
    print('eval batch num:%d'%eval_batch_num)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        # ckpt = tf.train.get_checkpoint_state(Load_Model_Path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCHS):
            for j in range(train_batch_num):
                X_batch, Y_batch = next(train_iter)
                feed_dict = {
                    train_model.xs: X_batch,
                    train_model.ys: Y_batch
                    }
                _, cost = sess.run([train_model.train_op, train_model.cost],
                                   feed_dict=feed_dict)

                if j % 100 == 0:
                    print("epoch:" + str(i) + ", train step: " + str(j) + ", loss: " + str(cost))
            if i % 10 == 0:
                print("evaluate after " + str(i) + " epoch:")
                run_eval(sess, eval_model, eval_data, eval_batch_num)
                save_path = os.path.join(MODEL_SAVE_PATH, "train" + str(i) + ".ckpt")
                saver.save(sess, save_path)
        print("evaluate after trainging")
        run_eval(sess, eval_model, eval_data, eval_batch_num)
        save_path = os.path.join(MODEL_SAVE_PATH, "aftrain.ckpt")
        saver.save(sess, save_path)

main()

                



