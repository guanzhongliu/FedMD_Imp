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
    conf_file = os.path.abspath("conf/pretrain_MNIST_conf.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
        return conf_file


models = {"2_layer_CNN": cnn_2layer_fc_model,
          "3_layer_CNN": cnn_3layer_fc_model}


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
        print(resulting_val_acc)

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
