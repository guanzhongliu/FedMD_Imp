import argparse
import errno
import os
import pickle
import sys

from tensorflow.keras.callbacks import EarlyStopping
from Neural_Networks import cnn_2layer_fc_model, cnn_3layer_fc_model
from load_data import load_MNIST_data


def parseArg():
    # 创建一个解析器
    parser = argparse.ArgumentParser()
    # metavar - 在使用方法消息中使用的参数值示例。
    parser.add_argument('-conf', metavar='conf.file', nargs=1,
                        help='Please choose the config file for training')
    conf_file = os.path.abspath("conf/pre_train.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
        return conf_file


models = {"2_layer_CNN": cnn_2layer_fc_model,
          "3_layer_CNN": cnn_3layer_fc_model}

# EarlyStopping允许设置的终止训练的条件即参数如下：
#
# monitor：监控的数据接口。keras定义了如下的数据接口可以直接使用：
# patient：对于设置的monitor，可以忍受在多少个epoch内没有改进？
# min_delta：评判monitor是否有改进的标准，只有变动范围大于min_delta的monitor才算是改进。对于连续在patient个epoch内没有改进的情况执行EarlyStopping。
# mode：只有三种情况{‘min’,’max’,’auto’}，分别表示monitor正常情况下是上升还是下降。比如当monitor为acc时mode要设置为’max’，因为正确率越大越好，相反，当monitor为loss时mode要设置为’min’。
# verbose：是否输出更多的调试信息。
# baseline：monitor的基线，即当monitor在基线以上没有改进时EarlyStopping。
# restore_best_weights：当发生EarlyStopping时，模型的参数未必是最优的，即monitor的指标未必处于最优状态。如果restore_best_weights设置为True，则自动查找最优的monitor指标时的模型参数。
def train_models(models, X_train, y_train, X_test, y_test,
                 is_show=False, save_dir="./", save_names=None,
                 early_stopping=True,
                 min_delta=0.001, patience=3, batch_size=128, epochs=20, is_shuffle=True, verbose=1,
                 ):
    '''
    Train an array of models on the same dataset.
    We use early termination to speed up training.
    '''

    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        if early_stopping:
            model.fit(X_train, y_train,
                      validation_data=[X_test, y_test],
                      callbacks=[EarlyStopping(monitor='accuracy', min_delta=min_delta, patience=patience)],
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )
        else:
            model.fit(X_train, y_train,
                      validation_data=[X_test, y_test],
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )

        resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        record_result.append({"train_acc": model.history.history["accuracy"],
                              "val_acc": model.history.history["val_accuracy"],
                              "train_loss": model.history.history["loss"],
                              "val_loss": model.history.history["val_loss"]})

        save_dir_path = os.path.abspath(save_dir)
        # make dir
        try:
            os.makedirs(save_dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if save_names is None:
            file_name = save_dir + "model_{0}".format(n) + ".h5"
        else:
            file_name = save_dir + save_names[n] + ".h5"
        model.save(file_name)

    if is_show:
        print("pre-train accuracy: ")
        for mm in resulting_val_acc:
            print(mm)

    return record_result


if __name__ == "__main__":
    # 解析所有传入的参数
    conf_file = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())
    dataset = conf_dict["data_type"]
    # 需要分类的数量
    n_classes = conf_dict["n_classes"]
    # 拿到所有的模型结构
    model_config = conf_dict["models"]
    train_params = conf_dict["train_params"]
    save_dir = conf_dict["save_directory"]
    save_names = conf_dict["save_names"]
    early_stopping = conf_dict["early_stopping"]

    del conf_dict

    # 选择数据集
    if dataset == "MNIST":
        input_shape = (28, 28)
        X_train, y_train, X_test, y_test = load_MNIST_data(standarized=True,
                                                           verbose=True)
    else:
        print("Unknown dataset. Program stopped.")
        sys.exit()

    pretrain_models = []

    # 迭代模型结构
    for i, item in enumerate(model_config):
        # 获取模型名称
        name = item["model_type"]
        # 获取模型的具体参数
        model_params = item["params"]

        tmp = models[name](n_classes=n_classes,
                           input_shape=input_shape,
                           **model_params)  # 以字典方式传入参数

        print("model {0} : {1}".format(i, save_names[i]))
        print(tmp.summary())
        pretrain_models.append(tmp)

    record_result = train_models(pretrain_models, X_train, y_train, X_test, y_test,
                                 save_dir=save_dir, save_names=save_names, is_show=True,
                                 early_stopping=early_stopping,
                                 **train_params)
    with open('pretrain_result.pkl', 'wb') as f:
        pickle.dump(record_result, f, protocol=pickle.HIGHEST_PROTOCOL)