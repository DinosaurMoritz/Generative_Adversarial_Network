from resources import *
import copy


def get_mnist():
    with open("mnist.json") as f:
        return json.load(f)


class Neural_network:

    def __init__(self, *args, activation_bundle=None, learning_rate=0.1):
        if not args:
            raise ValueError("Specify input, hidden and output Layer sizes!")
        if activation_bundle is None:
            activation_bundle = sigmoid, d_sigmoid
        self.activation_function, self.d_activation_function = activation_bundle

        self.args = args
        self.learning_rate = learning_rate

        self.number_of_inputs = args[0]
        self.number_of_Layers = len(args) - 1
        self.number_of_Layers_minus_1 = self.number_of_Layers - 1

        self.layer_neuron_numbers = [args[n + 1] for n in range(self.number_of_Layers)]

        self.network = self.generate_network()

        self.forward = self.feed_forward

    def generate_network(self):
        network = []
        number_of_inputs = self.number_of_inputs
        for number_of_neurons_in_layer in self.layer_neuron_numbers:
            layer = []
            for neuron_number in range(number_of_neurons_in_layer):
                layer.append([random.random() - 0.5 for _ in range(number_of_inputs + 1)])
            network.append(layer)
            number_of_inputs = number_of_neurons_in_layer
        return network

    def load(self, file_name="Neural_network"):
        with open(file_name + ".json", "r") as f:
            self.network = json.load(f)

    def save(self, file_name="Neural_network"):
        with open(file_name + ".json", "w") as f:
            json.dump(self.network, f)

    def feed_forward(self, inputs):
        for layer in self.network:
            layer_output = []
            for neuron in layer:
                neuron_output = sum([weight * inp for weight, inp in zip(neuron, inputs)]) + neuron[-1]
                neuron_activation = self.activation_function(neuron_output)
                layer_output.append(neuron_activation)

            inputs = layer_output

        return inputs

    def train_batch(self, batch):

        new_network = copy.deepcopy(self.network)

        for training_inputs, training_outputs in batch:
            # print(training_inputs)
            # time.sleep(0.001)

            all_layer_outputs = []
            all_layer_inputs = []
            for layer in self.network:
                all_layer_inputs.append(training_inputs)
                layer_output = []
                for neuron in layer:
                    neuron_output = sum([inp * weight for inp, weight in zip(training_inputs, neuron)]) + neuron[-1]
                    neuron_activation = self.activation_function(neuron_output)
                    layer_output.append(neuron_activation)
                training_inputs = layer_output
                all_layer_outputs.append(layer_output)
            network_output = training_inputs

            # print(network_output)

            # OUTPUT NEURON DELTAS
            all_neuron_deltas = [
                [-(target_output - output) * self.d_activation_function(output) for output, target_output in
                 zip(network_output, training_outputs)]]

            # HIDDEN DELTAS
            for layer_number in range(self.number_of_Layers_minus_1):  # enumerate(all_layer_outputs):
                real_layer_number = -(layer_number + 2)
                num_of_neurons_in_layer = self.layer_neuron_numbers[real_layer_number]
                layer_deltas = []

                layer_outputs = all_layer_outputs[real_layer_number]

                for a in range(num_of_neurons_in_layer):
                    neuron_error = 0
                    shallower_layer_number = real_layer_number + 1
                    shallower_layer = self.network[shallower_layer_number]
                    num_of_neurons_in_shallower_layer = self.layer_neuron_numbers[shallower_layer_number]

                    for b in range(num_of_neurons_in_shallower_layer):
                        neuron_error += shallower_layer[b][a] * all_neuron_deltas[-1][b]

                    layer_deltas.append(neuron_error * self.d_activation_function(layer_outputs[a]))

                all_neuron_deltas.append(layer_deltas)

            # print(all_neuron_deltas)
            # time.sleep(1)
            # print("-" *80)
            # continue

            # UPDATE NEURON WEIGHTS
            for layer_number in range(self.number_of_Layers):
                real_layer_number = -(layer_number + 1)
                num_of_neurons_in_layer = self.layer_neuron_numbers[real_layer_number]
                layer_inputs = all_layer_inputs[real_layer_number]
                for neuron_number in range(num_of_neurons_in_layer):
                    for weight_number in range(self.args[real_layer_number - 1]):
                        weight_error = all_neuron_deltas[layer_number][neuron_number] * layer_inputs[weight_number]
                        # print(weight_error * self.learning_rate)
                        new_network[real_layer_number][neuron_number][
                            weight_number] -= weight_error * self.learning_rate
                        # print(weight_error * self.learning_rate)

                    # BIAS
                    new_network[real_layer_number][neuron_number][-1] -= all_neuron_deltas[layer_number][
                                                                             neuron_number] * self.learning_rate
                    # print(all_neuron_deltas[layer_number][neuron_number] * self.learning_rate)

        self.network = new_network

    def train(self, ts, epochs=1000, batch_size=20):
        print("Starting training!")
        t = time.time()
        if callable(ts):
            for _ in range(epochs):
                for batch in lift(ts(), batch_size):
                    self.train_batch(batch)
        elif isinstance(ts, list):
            ts = lift(ts, batch_size)
            for _ in range(epochs):
                for batch in ts:
                    self.train_batch(batch)
        else:
            raise ValueError("Weird training set format!")
        print(f"Done training in {time.time() - t}s!")
        nn.save()


if __name__ == "__main__":
    import filecmp

    print(filecmp.cmp("comp_new.txt", "comp_old.txt", shallow=False))
    exit()


    nn = Neural_network(28*28, 32, 10)
    mnist = get_mnist()[:1000]

    # nn.load()
    nn.train(mnist, 1, 32)
    # nn.train(mnist, 1, 32)

    # nn.save("after")

    with open("comp_new.txt", "w") as f:
        f.write(str(nn.network))

    # print(nn.forward([random.random() for _ in range(28*28)]))

    print()