import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

from Neural_Networks import remove_last_layer
from load_data import generate_alignment_data


class FedMD():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize, train_type, interference=False, in_model=0,
                 random_logits=[]):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []
        self.init_result = []
        self.train_type = train_type
        self.in_model = in_model
        self.interference = interference
        self.random_logits = random_logits
        self.cosine_weights = []

        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=32, epochs=25, shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='accuracy', min_delta=0.001, patience=5)]
                             )

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})

            self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
                                     "train_acc": model_A_twin.history.history['accuracy'],
                                     "val_loss": model_A_twin.history.history['val_loss'],
                                     "train_loss": model_A_twin.history.history['loss'],
                                     })

            print()
            del model_A, model_A_twin
        # END FOR LOOP

        print("calculate the theoretical upper bounds for participants: ")

        self.upper_bounds = []
        self.pooled_train_result = []
        for model in parties:
            model_ub = clone_model(model)
            model_ub.set_weights(model.get_weights())
            model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                             loss="sparse_categorical_crossentropy",
                             metrics=["acc"])

            model_ub.fit(total_private_data["X"], total_private_data["y"],
                         batch_size=32, epochs=50, shuffle=True, verbose=1,
                         validation_data=[private_test_data["X"], private_test_data["y"]],
                         callbacks=[EarlyStopping(monitor='acc', min_delta=0.001, patience=5)])

            self.upper_bounds.append(model_ub.history.history["val_acc"][-1])
            self.pooled_train_result.append({"val_acc": model_ub.history.history["val_acc"],
                                             "acc": model_ub.history.history["acc"]})

            del model_ub
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)

    def collaborative_training_normal(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        save_rand = []
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0

            save_logits = []
            for nn in range(self.N_parties):
                d = self.collaborative_parties[nn]
                d["model_logits"].set_weights(d["model_weights"])
                save_logits.append(d["model_logits"].predict(alignment_data["X"], verbose=0))

                if self.interference is True and nn == self.in_model:
                    if self.random_logits == []:
                        shape = save_logits[nn].shape
                        temp = np.random.rand(shape[0], shape[1])
                        num = np.sum(temp)
                        temp = temp / num
                    else:
                        temp = self.random_logits[r]

                    save_logits[nn] = temp
                    save_rand.append(temp)

                logits += save_logits[nn]

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                print(collaboration_performance[index][-1])
                del y_pred

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], logits,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)
        return collaboration_performance, save_rand

    def collaborative_training_own(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)
            print("update logits ... ")
            # update logits
            logits = 0
            ans = []
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                temp = d["model_logits"].predict(alignment_data["X"], verbose=0)
                logits += temp
                ans.append(temp)

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))

                print(collaboration_performance[index][-1])
                del y_pred

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], ans[index],
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)
        return collaboration_performance

    def collaborative_training_simu(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        save_rand = []
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0
            save_logits = []
            for nn in range(self.N_parties):
                d = self.collaborative_parties[nn]
                d["model_logits"].set_weights(d["model_weights"])
                save_logits.append(d["model_logits"].predict(alignment_data["X"], verbose=0))
                if self.interference is True and nn == self.in_model:
                    if self.random_logits == []:
                        shape = save_logits[nn].shape
                        temp = np.random.rand(shape[0], shape[1])
                        num = np.sum(temp)
                        temp = temp / num
                    else:
                        temp = self.random_logits[r]

                    save_logits[nn] = temp
                    save_rand.append(temp)

                logits += save_logits[nn]

            logits /= self.N_parties

            cosine_simu = []
            for nn in range(self.N_parties):
                n = np.multiply(logits, save_logits[nn])
                denom = np.linalg.norm(logits) * np.linalg.norm(save_logits[nn])
                num = np.sum(n)
                cos = num / denom
                cosine_simu.append(0.5 + 0.5 * cos)
            x = np.array(cosine_simu)

            x = np.exp(x) / np.sum(np.exp(x), axis=0)

            self.cosine_weights.append(x)

            logits = 0
            for i in range(self.N_parties):
                logits += self.collaborative_parties[i]["model_logits"].predict(alignment_data["X"], verbose=0) * x[i]

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))

                print(collaboration_performance[index][-1])
                del y_pred

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], logits,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)
        return collaboration_performance, save_rand


class FedMD_random():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize,
                 random_parties, train_type, interference=False, in_model=0):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        self.random_parties = random_parties
        self.collaborative_parties = []
        self.init_result = []
        self.train_type = train_type

        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=32, epochs=25, shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='accuracy', min_delta=0.001, patience=5)]
                             )

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})

            self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
                                     "train_acc": model_A_twin.history.history['accuracy'],
                                     "val_loss": model_A_twin.history.history['val_loss'],
                                     "train_loss": model_A_twin.history.history['loss'],
                                     })

            print()
            del model_A, model_A_twin
        # END FOR LOOP

        print("calculate the theoretical upper bounds for participants: ")

        self.upper_bounds = []
        self.pooled_train_result = []
        for model in parties:
            model_ub = clone_model(model)
            model_ub.set_weights(model.get_weights())
            model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                             loss="sparse_categorical_crossentropy",
                             metrics=["acc"])

            model_ub.fit(total_private_data["X"], total_private_data["y"],
                         batch_size=32, epochs=50, shuffle=True, verbose=1,
                         validation_data=[private_test_data["X"], private_test_data["y"]],
                         callbacks=[EarlyStopping(monitor='acc', min_delta=0.001, patience=5)])

            self.upper_bounds.append(model_ub.history.history["val_acc"][-1])
            self.pooled_train_result.append({"val_acc": model_ub.history.history["val_acc"],
                                             "acc": model_ub.history.history["acc"]})

            del model_ub
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0

            slice = random.sample(self.collaborative_parties, self.random_parties)

            for d in slice:
                d["model_logits"].set_weights(d["model_weights"])
                logits += d["model_logits"].predict(alignment_data["X"], verbose=0)

            logits /= self.random_parties + self.train_type

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))

                print(collaboration_performance[index][-1])
                del y_pred

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], logits,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        print("the upper bounds are:")
        for i in self.upper_bounds:
            print(i)
        return collaboration_performance
