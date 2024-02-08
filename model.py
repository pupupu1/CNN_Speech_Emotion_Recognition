### experiment reproducibility ###
from tensorflow import keras

seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


import numpy as np
import tensorflow.keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import model_from_json
from sklearn.metrics import classification_report, confusion_matrix, normalized_mutual_info_score, plot_confusion_matrix
from keras.utils.vis_utils import plot_model
import seaborn
import matplotlib
import preprocessing as pre_proc
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

matplotlib.rc("font",family='SimHei') # 显示中文
### enable/disable attention ###
ENABLE_ATTENTION = True


def create_model(units=256):
    input = keras.Input(shape=(pre_proc.N_FRAMES, pre_proc.N_FEATURES))
    if MODEL == "Attention_BLSTM":
        states, forward_h, _, backward_h, _ = layers.Bidirectional(
            #return_sequences: 返回单个 hidden state值还是返回全部time step 的 hidden state值
            #return_state: 是否返回除输出之外的最后一个状态
            layers.LSTM(units, return_sequences=True, return_state=True)  #输出的hidden state包含全部时间步的结果
        )(input)
        last_state = layers.Concatenate()([forward_h, backward_h])  #Concatenate拼接
        #layers.Dense定义网络层
        #选择tanh作为激活函数；use_bias参数的作用就是决定该卷积层输出是否有偏移量b
        hidden = layers.Dense(units, activation="tanh", use_bias=False,
                              #权重矩阵初始化方法：RandomNormal正态分布初始化 均值0 标准差1
                              kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.)
                              )(states)
        #线性激活函数
        out = layers.Dense(1, activation='linear', use_bias=False,
                              kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.)
                              )(hidden)
        #flatten用于将输入层的数据压成一维的数据，一般用再卷积层和全连接层之间（全连接层只能接收一维数据，而卷积层可以处理二维数据）
        flat = layers.Flatten()(out)
        #lambda匿名函数层
        energy = layers.Lambda(lambda x:x/np.sqrt(units))(flat)
        #softmax层 （多分类时的归一化函数） 正规化 转换成概率
        normalize = layers.Softmax()
        normalize._init_set_name("alpha")
        alpha = normalize(energy)
        #上下文向量； Dot：batch中每一组对应的样本之间进行点乘
        context_vector = layers.Dot(axes=1)([states, alpha])
        context_vector = layers.Concatenate()([context_vector, last_state])
    elif MODEL == "BLSTM":
        context_vector = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(input)
    else:
        raise Exception("Unknown model architecture!")
    pred = layers.Dense(pre_proc.N_EMOTIONS, activation="softmax")(context_vector)
    model = keras.Model(inputs=[input], outputs=[pred])
    model._init_set_name(MODEL)
    print(str(model.summary()))
    return model


def train_and_test_model(model):
    X_train, X_test, y_train, y_test = pre_proc.get_train_test()
    #categorical_crossentropy多类（多个标签时使用）交叉熵
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    plot_model(model, MODEL+"_model.png", show_shapes=True)
    best_weights_file = MODEL+"_weights.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs= 40,
        batch_size=32,
        callbacks=[es,mc],
        verbose=2
    )
    save(model)#提前终止训练过程
    # model testing
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if MODEL == "Attention_BLSTM":
        plt.title('模型准确率  CNN-BiLSTM+注意力')
    else:
        plt.title('模型准确率  CNN-BiLSTM')  #model accuracy - BLSTM without attention
    plt.ylabel('准确率')
    plt.xlabel('epoch')
    plt.legend(['训练', '测试'], loc='upper left')
    plt.savefig(MODEL+"_accuracy.png")
    plt.gcf().clear()  # clear
    # loss on validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if MODEL == "Attention_BLSTM":
        plt.title('模型损失  CNN-BiLSTM+注意力')
    else:
        plt.title('模型损失  CNN-BiLSTM')  #model loss - BLSTM without attention
    plt.ylabel('损失')
    plt.xlabel('epoch')
    plt.legend(['训练', '测试'], loc='upper left')
    plt.savefig(MODEL+"_loss.png")
    plt.gcf().clear()  # clear
    # test acc and loss
    model.load_weights(best_weights_file) # load the best saved model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    test_metrics = model.evaluate(X_test, y_test, batch_size=32)
    print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
    print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
    print("测试准确率: " + str(format(test_metrics[1], '.3f')) + "\n")
    print("测试损失: " + str(format(test_metrics[0], '.3f')) + "\n")
    # test acc and loss per class
    real_class = np.argmax(y_test, axis=1)
    pred_class_probs = model.predict(X_test)
    pred_class = np.argmax(pred_class_probs, axis=1)
    target_names = pre_proc.emo_labels_en

    report = classification_report(real_class, pred_class, )
    print("分类结果:\n" + str(report) + "\n")
    print(classification_report(real_class, pred_class, target_names=target_names))

    cm = confusion_matrix(real_class, pred_class)   #confusion_matrix

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')

    print("混淆矩阵:\n" + str(cm) + "\n")
    data = np.array([value for value in cm_normalized.flatten()]).reshape(7,7)
    if MODEL == "Attention_BLSTM":
        plt.title('CNN-BiLSTM+注意力')
    else:
        plt.title('CNN-BiLSTM')  #BLSTM without attention
    seaborn.heatmap(cm_normalized, xticklabels=pre_proc.emo_labels_en, yticklabels=pre_proc.emo_labels_en, annot=data, cmap="Reds")
    plt.savefig(MODEL+"_conf_matrix.png")


def visualize_attention(model):
    best_weights_file = MODEL + "_weights.h5"
    model.load_weights(best_weights_file)
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    _, X_test, _, _ = pre_proc.get_train_test()
    predictions = model.predict(X_test)
    labels = np.argmax(predictions, axis=1)
    # inspect attention weigths
    attention = model.get_layer(name="alpha")
    weigth_model = keras.Model(inputs=model.input, outputs=attention.output)
    attention_weights = weigth_model.predict(X_test)
    d = {}
    for w, l in zip(attention_weights, labels):
        if l not in d:
            d[l] = w
        else:
            d[l] += w
    data = []
    for x, y in d.items():
        norm_w = y / np.sum(y)
        data.append(norm_w)
    #reshape and trim
    bins = 10
    bin_c = pre_proc.N_FRAMES//bins
    trim = pre_proc.N_FRAMES%bins
    data = np.asarray(data).reshape(pre_proc.N_EMOTIONS, pre_proc.N_FRAMES)[:, trim:]
    data = np.sum(data.reshape([7, bins, bin_c]), axis=2).reshape(pre_proc.N_EMOTIONS,bins)
    plt.clf()
    seaborn.heatmap(data, yticklabels=pre_proc.emo_labels_en, cmap="Reds")
    plt.savefig("visualize_attention.png")


def load():
    with open("model.json", 'r') as f:
        model = model_from_json(f.read())
    best_weights_file = MODEL + "_weights.h5"
    # Load weights into the new model
    model.load_weights(best_weights_file)
    return model


def save(model):
    model_json = model.to_json()
    with open(MODEL+"_model.json", "w") as json_file:
        json_file.write(model_json)
    print("model saved")




######### SPEECH EMOTION RECOGNITION #########

# 1) feature extraction
pre_proc.feature_extraction()

# 2) select model
if ENABLE_ATTENTION:
    MODEL = "Attention_BLSTM"
else:
    MODEL = "BLSTM"

# 3) create model
model = create_model()

# 4) train and test model
train_and_test_model(model)

# 5) visualize attention weights
if ENABLE_ATTENTION:
    visualize_attention(model)