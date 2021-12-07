from resources import *
from Neural_Network import Neural_network


# def get_mnist_without_label():
#     with open("mnist_without_label.json") as f:
#         return json.load(f)


class GAN:

    def __init__(self, generator=None, discriminator=None, generator_activation_bundle=None,
                 discriminator_activation_bundle=None):
        if generator is None or discriminator is None:
            raise ValueError("Specify input, hidden and output Layer sizes!")

        self.generator_args = generator
        self.discriminator_args = discriminator

        self.number_of_Layers_in_GAN = len(self.generator_args) - 1 + len(self.discriminator_args) - 1
        self.number_of_Layers_minus_1 = self.number_of_Layers_in_GAN - 1

        self.number_of_Layers_in_generator = len(self.generator_args) - 1

        self.layer_neuron_numbers_of_GAN = [self.generator_args[n + 1] for n in range(len(self.generator_args) - 1)] + \
                                           [self.discriminator_args[n + 1] for n in
                                            range(len(self.discriminator_args) - 1)]

        self.generator_learning_rate = 0.2
        self.discriminator_learning_rate = 0.1

        self.generator_training_cycles = 20

        self.generator = Neural_network(*generator, activation_bundle=generator_activation_bundle)
        self.discriminator = Neural_network(*discriminator, activation_bundle=discriminator_activation_bundle,
                                            learning_rate=self.discriminator_learning_rate)

        self.generator_activation_function, self.generator_d_activation_function = self.generator.activation_function, self.generator.d_activation_function

        self.real = [0]
        self.fake = [1]

        self.generate_image = self.generator.forward

    def save(self):
        self.generator.save("generator_new")
        self.discriminator.save("discriminator_new")

    def load(self):
        self.generator.load("generator_new")
        self.discriminator.load("discriminator_new")

    def error(self, ts):
        errors = []
        for inp in ts:
            if random.random() > 0.5:
                res = self.discriminator.forward(inp)
                tg = self.real
            else:
                res = self.discriminator.forward(self.generator.forward([random.random() for _ in range(100)]))
                tg = self.fake
            errors.append(sum([0.5 * pow(a - b, 2) for a, b in zip(tg, res)]))
        return sum(errors) / len(errors)

    def sum_network(self):
        return sum(flatten(flatten(self.discriminator.network))) + sum(flatten(flatten(self.generator.network)))

    def train_batch(self, batch):
        discriminator_batch = []

        for ex in batch:
            if random.random() > 0.5:
                discriminator_batch.append((self.generator.forward([random.random() for _ in range(100)]), self.fake))
            else:
                discriminator_batch.append([ex, self.real])

        self.discriminator.train_batch(discriminator_batch)

        for _ in range(self.generator_training_cycles):
            new_network = copy.deepcopy(self.generator.network)

            for _ in batch:
                combined_layers = self.generator.network + self.discriminator.network

                training_inputs = [random.random() for _ in range(100)]

                # FORWARD PROPAGATION
                all_layer_outputs = []
                all_layer_inputs = []
                for layer in combined_layers:
                    all_layer_inputs.append(training_inputs)
                    layer_output = []
                    for neuron in layer:
                        neuron_output = sum([inp * weight for inp, weight in zip(training_inputs, neuron)]) + neuron[-1]
                        neuron_activation = self.generator_activation_function(neuron_output)
                        layer_output.append(neuron_activation)
                    training_inputs = layer_output
                    all_layer_outputs.append(layer_output)
                network_output = training_inputs

                # OUTPUT NEURON DELTAS
                all_neuron_deltas = [
                    [-(target_output - output) * self.generator_d_activation_function(output) for output, target_output
                     in
                     zip(network_output, self.real)]]

                # HIDDEN DELTAS
                for layer_number in range(self.number_of_Layers_minus_1):
                    real_layer_number = -(layer_number + 2)
                    num_of_neurons_in_layer = self.layer_neuron_numbers_of_GAN[real_layer_number]
                    layer_deltas = []

                    layer_outputs = all_layer_outputs[real_layer_number]

                    for a in range(num_of_neurons_in_layer):
                        neuron_error = 0
                        shallower_layer_number = real_layer_number + 1
                        shallower_layer = combined_layers[shallower_layer_number]
                        num_of_neurons_in_shallower_layer = self.layer_neuron_numbers_of_GAN[shallower_layer_number]

                        for b in range(num_of_neurons_in_shallower_layer):
                            neuron_error += shallower_layer[b][a] * all_neuron_deltas[-1][b]

                        layer_deltas.append(neuron_error * self.generator_d_activation_function(layer_outputs[a]))

                    all_neuron_deltas.append(layer_deltas)

                generator_deltas = all_neuron_deltas[-self.number_of_Layers_in_generator:]

                # UPDATE NEURON WEIGHTS
                for layer_number in range(self.generator.number_of_Layers):
                    real_layer_number = -(layer_number + 1)

                    num_of_neurons_in_layer = self.generator_args[real_layer_number]
                    number_of_inputs_to_layer = self.generator_args[real_layer_number - 1]
                    layer_inputs = all_layer_inputs[real_layer_number - 2]

                    for neuron_number in range(num_of_neurons_in_layer):
                        for weight_number in range(number_of_inputs_to_layer):
                            weight_error = generator_deltas[layer_number][neuron_number] * \
                                           layer_inputs[weight_number]
                            new_network[real_layer_number][neuron_number][
                                weight_number] -= weight_error * self.generator_learning_rate

                        # BIAS
                        new_network[real_layer_number][neuron_number][-1] -= generator_deltas[layer_number][
                                                                                 neuron_number] * self.generator_learning_rate

            self.generator.network = new_network


if __name__ == "__main__":
    gan = GAN([100, 128, 28 * 28], [28 * 28, 32, 32, 1], generator_activation_bundle=[tanh, d_tanh])
    gan.load()


    # combined_layers = gan.generator.network + gan.discriminator.network
    #
    # training_inputs = [random.random() for _ in range(100)]
    #
    # # FORWARD PROPAGATION
    # for layer in combined_layers:
    #     layer_output = []
    #     for neuron in layer:
    #         neuron_output = sum([inp * weight for inp, weight in zip(training_inputs, neuron)]) + neuron[-1]
    #         neuron_activation = gan.generator_activation_function(neuron_output)
    #         layer_output.append(neuron_activation)
    #     training_inputs = layer_output
    # network_output = training_inputs
    # print(network_output)
    #
    # exit()

    mnist = lift(get_mnist_without_label(), 64)

    print("Starting training!")
    c = 0
    while True:
        for i, monster_batch in enumerate(lift(mnist, 10)):
            for batch in monster_batch:
                gan.train_batch(batch)

            print("Done monster batch", i)
            gan.save()
            print("Error : ", gan.error(monster_batch[0]))
            turn_to_pic([x * 255 for x in gan.generator.forward([random.random() for _ in range(100)])],
                        name=f"pic{i}.png")
            gan.save()
