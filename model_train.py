import argparse
import errno
import os
import pickle
import sys
import numpy as np
from tensorflow.keras.models import load_model
from FedMD import FedMD, FedMD_random
from load_data import load_MNIST_data, load_EMNIST_data, generate_bal_private_data, \
    generate_partial_data


def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for FedMD.'
                        )

    conf_file = os.path.abspath("conf/ac_train.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


if __name__ == "__main__":
    conf_file = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        emnist_data_dir = conf_dict["EMNIST_dir"]
        N_parties = conf_dict["N_parties"]
        private_classes = conf_dict["private_classes"]
        N_samples_per_class = conf_dict["N_samples_per_class"]
        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]
        model_saved_dir = conf_dict["model_saved_dir"]
        random_parties = conf_dict["Random_parties"]
        result_save_dir = conf_dict["result_save_dir"]
        train_type = conf_dict["train_type"]
        interference = conf_dict["interference"]

    del conf_dict, conf_file

    X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST \
        = load_MNIST_data(standarized=True, verbose=True)

    public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}
    del X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST

    X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST \
        = load_EMNIST_data(emnist_data_dir,
                           standarized=True, verbose=True)

    # generate private data
    private_data, total_private_data \
        = generate_bal_private_data(X_train_EMNIST, y_train_EMNIST,
                                    N_parties=N_parties,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=False)

    X_tmp, y_tmp = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                         class_in_use=private_classes, verbose=True)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp

    if model_saved_dir is not None:
        parties = []
        dpath = os.path.abspath(model_saved_dir)
        model_names = os.listdir(dpath)
        for name in model_names:
            tmp = None
            tmp = load_model(os.path.join(dpath, name))
            parties.append(tmp)

    if interference is True:
        in_model = np.random.randint(0, high=10)
    else:
        in_model = 0

    if train_type == "contrast":
        fed_base = FedMD(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_alignment=N_alignment,
                        N_logits_matching_round=N_logits_matching_round,
                        logits_matching_batchsize=logits_matching_batchsize,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize, train_type=train_type)

        collaboration_performance_base = fed_base.collaborative_training_normal()

        fed_sim = FedMD(parties,
                        public_dataset=public_dataset,
                        private_data=private_data,
                        total_private_data=total_private_data,
                        private_test_data=private_test_data,
                        N_rounds=N_rounds,
                        N_alignment=N_alignment,
                        N_logits_matching_round=N_logits_matching_round,
                        logits_matching_batchsize=logits_matching_batchsize,
                        N_private_training_round=N_private_training_round,
                        private_training_batchsize=private_training_batchsize, train_type=train_type,
                        interference=interference, in_model=in_model)

        collaboration_performance_sim, save_rand = fed_sim.collaborative_training_simu()

        fedmd = FedMD(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data,
                      N_rounds=N_rounds,
                      N_alignment=N_alignment,
                      N_logits_matching_round=N_logits_matching_round,
                      logits_matching_batchsize=logits_matching_batchsize,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize, train_type=train_type,
                      interference=interference, in_model=in_model, random_logits=save_rand)

        collaboration_performance, save_rand = fedmd.collaborative_training_normal()

        fedmd_own = FedMD(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_alignment=N_alignment,
                          N_logits_matching_round=N_logits_matching_round,
                          logits_matching_batchsize=logits_matching_batchsize,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize, train_type=train_type,
                          interference=interference, in_model=in_model)

        collaboration_performance_own = fedmd_own.collaborative_training_own()

        print("downbound_results:")
        for i in range(N_parties):
            print(collaboration_performance_own[i][-1])

        print("inference model: {}".format(in_model))

        print("base_results:")
        for i in range(N_parties):
            print(collaboration_performance_base[i][-1])

        print("normal_results:")
        for i in range(N_parties):
            print(collaboration_performance[i][-1])

        print("cosine_results:")
        for i in range(N_parties):
            print(collaboration_performance_sim[i][-1])

        print("cosine_weights:")
        for i in fed_sim.cosine_weights:
            print(i)

    else:
        if train_type == "actual":
            fedmd = FedMD(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_alignment=N_alignment,
                          N_logits_matching_round=N_logits_matching_round,
                          logits_matching_batchsize=logits_matching_batchsize,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize, train_type=train_type)
            collaboration_performance = fedmd.collaborative_training_normal()

        elif train_type == "simu":
            fedmd = FedMD(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_alignment=N_alignment,
                          N_logits_matching_round=N_logits_matching_round,
                          logits_matching_batchsize=logits_matching_batchsize,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize, train_type=train_type)
            collaboration_performance = fedmd.collaborative_training_simu()

        elif train_type == "random":
            fedmd = FedMD_random(parties,
                                 public_dataset=public_dataset,
                                 private_data=private_data,
                                 total_private_data=total_private_data,
                                 private_test_data=private_test_data,
                                 N_rounds=N_rounds,
                                 N_alignment=N_alignment,
                                 N_logits_matching_round=N_logits_matching_round,
                                 logits_matching_batchsize=logits_matching_batchsize,
                                 N_private_training_round=N_private_training_round,
                                 private_training_batchsize=private_training_batchsize,
                                 random_parties=random_parties, train_type=train_type)
            collaboration_performance = fedmd.collaborative_training()

        elif train_type == "own":
            fedmd = FedMD(parties,
                          public_dataset=public_dataset,
                          private_data=private_data,
                          total_private_data=total_private_data,
                          private_test_data=private_test_data,
                          N_rounds=N_rounds,
                          N_alignment=N_alignment,
                          N_logits_matching_round=N_logits_matching_round,
                          logits_matching_batchsize=logits_matching_batchsize,
                          N_private_training_round=N_private_training_round,
                          private_training_batchsize=private_training_batchsize,
                          train_type=train_type)
            collaboration_performance = fedmd.collaborative_training_own()

        initialization_result = fedmd.init_result
        pooled_train_result = fedmd.pooled_train_result

        if result_save_dir is not None:
            save_dir_path = os.path.abspath(result_save_dir)
            # make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        with open(os.path.join(save_dir_path, 'init_result.pkl'), 'wb') as f:
            pickle.dump(initialization_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_dir_path, 'pooled_train_result.pkl'), 'wb') as f:
            pickle.dump(pooled_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_dir_path, 'col_performance.pkl'), 'wb') as f:
            pickle.dump(collaboration_performance, f, protocol=pickle.HIGHEST_PROTOCOL)
