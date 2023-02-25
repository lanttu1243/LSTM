import numpy as np
from matplotlib import pyplot as plt


class LSTMLayer:

    def __init__(self,
                 layer_number: int,
                 input_size: int,
                 hidden_size: int,
                 learning_rate: float,
                 output_size: int = 0):
        # Initialising the static parameters for the layer
        self.hidden_size: int = hidden_size  # Hidden size is the number of neurons in a layer
        self.input_size: int = input_size  # Input size is the number of inputs into the layer
        self.learning_rate: float = learning_rate  # Learning rate tells the model how quickly to adjust parameters
        self.smooth_loss: float = 0  # Initialising the loss to zero
        self.number: int = layer_number  # The number of the layer in the network
        # If the layer is the last layer this tells the size of
        # the output often the same as the input for the first layer
        self.output_size: int = output_size
        # define the variables used in the layer
        self.xs: dict[int, np.ndarray] = {}  # Input
        self.hs: dict[int, np.ndarray] = {}  # Hidden state
        self.fg: dict[int, np.ndarray] = {}  # Forget gate
        self.ig: dict[int, np.ndarray] = {}  # Input gate
        self.og: dict[int, np.ndarray] = {}  # Output gate
        self.mg: dict[int, np.ndarray] = {}  # Memory gate
        self.cs: dict[int, np.ndarray] = {}  # Cell gate
        self.ys: dict[int, np.ndarray] = {}  # Output
        self.ps: dict[int, np.ndarray] = {}  # Probability vector
        self.conc: dict[int, np.ndarray] = {}  # Concatenation of xs[t] and hs[t-1]

        # Initialize weights and biases for the lstm unit
        self.Wf: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.0001
        self.Wi: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.0001
        self.Wo: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.0001
        self.Wm: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.0001

        self.Bf: np.ndarray = np.zeros((self.hidden_size, 1))
        self.Bi: np.ndarray = np.zeros((self.hidden_size, 1))
        self.Bo: np.ndarray = np.zeros((self.hidden_size, 1))
        self.Bm: np.ndarray = np.zeros((self.hidden_size, 1))

        # initialize weights and biases for output layer
        self.Wy: np.ndarray = np.random.randn(self.output_size, self.hidden_size)
        self.By: np.ndarray = np.zeros((self.output_size, 1))

        # define variables to store the parameter updates to
        self.dWf: np.ndarray = np.zeros_like(self.Wf)
        self.dWi: np.ndarray = np.zeros_like(self.Wi)
        self.dWo: np.ndarray = np.zeros_like(self.Wo)
        self.dWm: np.ndarray = np.zeros_like(self.Wm)

        self.dBf: np.ndarray = np.zeros_like(self.Bf)
        self.dBi: np.ndarray = np.zeros_like(self.Bi)
        self.dBo: np.ndarray = np.zeros_like(self.Bo)
        self.dBm: np.ndarray = np.zeros_like(self.Bm)

        self.dWy: np.ndarray = np.zeros_like(self.Wy)
        self.dBy: np.ndarray = np.zeros_like(self.By)

        # define variables for Adagrad memories
        self.mWf: np.ndarray = np.zeros_like(self.Wf)
        self.mWi: np.ndarray = np.zeros_like(self.Wi)
        self.mWo: np.ndarray = np.zeros_like(self.Wo)
        self.mWm: np.ndarray = np.zeros_like(self.Wm)

        self.mBf: np.ndarray = np.zeros_like(self.Bf)
        self.mBi: np.ndarray = np.zeros_like(self.Bi)
        self.mBo: np.ndarray = np.zeros_like(self.Bo)
        self.mBm: np.ndarray = np.zeros_like(self.Bm)

        self.mWy: np.ndarray = np.zeros_like(self.Wy)
        self.mBy: np.ndarray = np.zeros_like(self.By)

        # define variables for derivative checking
        self.cWf: np.ndarray = np.zeros_like(self.Wf)
        self.cWi: np.ndarray = np.zeros_like(self.Wi)
        self.cWo: np.ndarray = np.zeros_like(self.Wo)
        self.cWm: np.ndarray = np.zeros_like(self.Wm)

        self.cBf: np.ndarray = np.zeros_like(self.Bf)
        self.cBi: np.ndarray = np.zeros_like(self.Bi)
        self.cBo: np.ndarray = np.zeros_like(self.Bo)
        self.cBm: np.ndarray = np.zeros_like(self.Bm)

        self.cWy: np.ndarray = np.zeros_like(self.Wy)
        self.cBy: np.ndarray = np.zeros_like(self.By)

        self.prev_hidden: np.ndarray = np.zeros((self.hidden_size, 1))
        self.prev_cell: np.ndarray = np.zeros((self.hidden_size, 1))

    def __str__(self):
        # Return the layer as a string
        if self.output_size != 0:
            out_size = self.output_size
        else:
            out_size = self.hidden_size
        return f"LSTM(Layer no. {self.number}, input size: {self.input_size}," \
               f"hidden size: {self.hidden_size}, output size: {out_size})"

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(- x, dtype="float64"))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - x * x

    def initialise_layer(self) -> None:
        # initialise the parameters for the iteration to empty or zero
        self.xs, self.hs, self.fg, self.ig, self.og, self.mg, self.cs = {}, {}, {}, {}, {}, {}, {}
        self.ys, self.ps = {}, {}
        self.conc = {}

        for i in [self.dWf, self.dWi, self.dWo, self.dWm, self.dBf, self.dBi, self.dBo, self.dBm, self.dWy, self.dBy]:
            i.fill(0)

    def format_text_input(self, inputs: list[int]) -> None:
        for t, k in enumerate(inputs):
            self.xs[t] = np.zeros((self.input_size, 1))
            self.xs[t][inputs[t]] = 1

    def format_non_text_input(self, inputs: dict[int, np.ndarray]) -> None:
        for t, k in enumerate(inputs):
            self.xs[t] = inputs[k]

    def forward_pass(self) -> dict[int, np.ndarray]:
        self.hs[-1] = self.prev_hidden
        self.cs[-1] = self.prev_cell
        for t, k in enumerate(self.xs):
            self.conc[t] = np.concatenate((self.xs[t], self.hs[t - 1]), axis=0)
            self.fg[t] = self.sigmoid(self.Wf @ self.conc[t] + self.Bf)
            self.ig[t] = self.sigmoid(self.Wi @ self.conc[t] + self.Bi)
            self.og[t] = self.sigmoid(self.Wo @ self.conc[t] + self.Bo)
            self.mg[t] = np.tanh(self.Wm @ self.conc[t] + self.Bm)

            self.cs[t] = self.fg[t] * self.cs[t - 1] + self.ig[t] * self.mg[t]
            self.hs[t] = self.og[t] * self.cs[t]
        self.prev_hidden = self.hs[len(self.xs) - 1]
        return self.hs

    def output_layer(self) -> dict[int, np.ndarray]:
        hid: dict[int, np.ndarray] = self.forward_pass().copy()
        hid.pop(-1)
        for t, k in enumerate(hid):
            self.ys[t] = self.Wy @ self.hs[t] + self.By
            yt = self.ys[t] - np.max(self.ys[t])
            self.ps[t] = np.exp(yt) / np.sum(np.exp(yt))
        return self.ps

    def backpropagation_output_layer(self, targets: list[int]) -> dict[int, np.ndarray]:
        dout: dict[int, np.ndarray] = {}
        for t in reversed(range(len(targets))):
            # Backpropagation into y
            dy = np.copy(self.ps[t])

            dy[targets[t]] -= 1

            self.dBy += dy
            self.dWy += dy @ self.hs[t].T

            dout[t] = self.Wy.T @ dy

        return dout

    def backpropagation_lstm_layer(self, dh: dict[int, np.ndarray]) -> dict[int, np.ndarray]:

        dcnext: np.ndarray = np.zeros((self.hidden_size, 1))
        dx: dict[int, np.ndarray] = {}
        for t in reversed(range(len(dh))):

            dout_dh = dh[t]
            dh_dct = dout_dh * self.og[t] + dcnext
            dh_do = self.cs[t]

            dcnext = dh_dct * self.fg[t]
            dct_df = dh_dct * self.cs[t-1]
            dct_di = dh_dct * self.mg[t]
            dct_dm = dh_dct * self.ig[t]

            df_dzf = dct_df * (self.fg[t] * (1 - self.fg[t]))
            self.dWf += df_dzf @ self.conc[t].T
            self.dBf += df_dzf
            dzf_dx = self.Wf.T @ df_dzf

            di_dzi = dct_di * (self.ig[t] * (1 - self.ig[t]))
            self.dWi += di_dzi @ self.conc[t].T
            self.dBi += di_dzi
            dzi_dx = self.Wi.T @ di_dzi

            dm_dzm = dct_dm * (1 - np.square(self.mg[t]))
            self.dWm += dm_dzm @ self.conc[t].T
            self.Bm += dm_dzm
            dzm_dx = self.Wm.T @ dm_dzm

            do_dzo = dh_do * (self.og[t] * (1 - self.og[t]))
            self.dWo += do_dzo @ self.conc[t].T
            self.Bo += do_dzo
            dzo_dx = self.Wo.T @ do_dzo

            dx[t] = (dzo_dx + dzm_dx + dzf_dx + dzi_dx)[:self.input_size, :]

        return dx

    def adagrad(self) -> None:
        self.mWf += np.square(self.dWf)
        self.mWi += np.square(self.dWi)
        self.mWo += np.square(self.dWo)
        self.mWm += np.square(self.dWm)

        self.mBf += np.square(self.dBf)
        self.mBi += np.square(self.dBi)
        self.mBo += np.square(self.dBo)
        self.mBm += np.square(self.dBm)

        self.mWy += np.square(self.dWy)
        self.mBy += np.square(self.dBy)

        self.Wf -= (self.learning_rate / np.sqrt(np.diagonal(self.mWf).copy() + 1e-8)) @ self.dWf
        self.Wi -= (self.learning_rate / np.sqrt(np.diagonal(self.mWi).copy() + 1e-8)) @ self.dWi
        self.Wo -= (self.learning_rate / np.sqrt(np.diagonal(self.mWo).copy() + 1e-8)) @ self.dWo
        self.Wm -= (self.learning_rate / np.sqrt(np.diagonal(self.mWm).copy() + 1e-8)) @ self.dWm

        self.Bf -= (self.learning_rate / np.sqrt(self.mBf + 1e-8)) * self.dBf
        self.Bi -= (self.learning_rate / np.sqrt(self.mBi + 1e-8)) * self.dBi
        self.Bo -= (self.learning_rate / np.sqrt(self.mBo + 1e-8)) * self.dBo
        self.Bm -= (self.learning_rate / np.sqrt(self.mBm + 1e-8)) * self.dBm

        self.Wy -= (self.learning_rate / np.sqrt(np.diagonal(self.mWy).copy() + 1e-8)) @ self.dWy
        self.By -= (self.learning_rate / np.sqrt(self.mBy + 1e-8)) * self.dBy

    def sgd(self) -> None:
        self.Wf -= self.learning_rate * self.dWf
        self.Wi -= self.learning_rate * self.dWi
        self.Wo -= self.learning_rate * self.dWo
        self.Wm -= self.learning_rate * self.dWm

        self.Bf -= self.learning_rate * self.dBf
        self.Bi -= self.learning_rate * self.dBi
        self.Bo -= self.learning_rate * self.dBo
        self.Bm -= self.learning_rate * self.dBm

        self.Wy -= self.learning_rate * self.dWy
        self.By -= self.learning_rate * self.dBy

    def loss(self, targets: list[int]) -> float:
        # Calculate Loss
        loss = 0
        for t in range(len(targets)):
            loss += - np.log(self.ps[t][targets[t]] + 1e-5)
        self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss
        return self.smooth_loss

    def forward_check(self, wf, wi, wo, wm, bf, bi, bo, bm) -> dict[int, np.ndarray]:
        hs, cs, fg, ig, og, mg, cs, = {}, {}, {}, {}, {}, {}, {}
        # Initialise h_t-1 and c_t-1 to zero-arrays
        hs[-1] = self.prev_hidden
        cs[-1] = self.prev_cell
        # forward pass with one variable slightly changed
        for t, k in enumerate(self.xs):
            # No need to initialise self.conc as it exists from forward pass
            fg[t] = self.sigmoid(wf @ self.conc[t] + bf)
            ig[t] = self.sigmoid(wi @ self.conc[t] + bi)
            og[t] = self.sigmoid(wo @ self.conc[t] + bo)
            mg[t] = np.tanh(wm @ self.conc[t] + bm)

            cs[t] = fg[t] * cs[t - 1] + ig[t] * mg[t]
            hs[t] = og[t] * cs[t]
        hs.pop(-1)
        return hs

    def output_check(self, wy, by) -> dict[int, np.ndarray]:
        hid: dict[int, np.ndarray] = self.forward_pass().copy()
        ys, ps = {}, {}
        hid.pop(-1)
        hs = hid
        for t, k in enumerate(hid):
            ys[t] = wy @ hs[t] + by
            yt = ys[t] - np.max(ys[t])
            ps[t] = np.exp(yt) / np.sum(np.exp(yt))
        return ps

    def grad_check(self) -> float:
        # Function for calculating the hidden states to check the derivatives
        def h(variable: str = "default", d=1e-20) -> dict[int, np.ndarray]:
            match variable.lower():
                case "wf":
                    return self.forward_check(wf=self.Wf + d, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf,
                                              bi=self.Bi, bo=self.Bo, bm=self.Bm)
                case "wi":
                    return self.forward_check(wf=self.Wf, wi=self.Wi + d, wo=self.Wo, wm=self.Wm, bf=self.Bf,
                                              bi=self.Bi, bo=self.Bo, bm=self.Bm)
                case "wo":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo + d, wm=self.Wm, bf=self.Bf,
                                              bi=self.Bi, bo=self.Bo, bm=self.Bm)
                case "wm":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm + d, bf=self.Bf,
                                              bi=self.Bi, bo=self.Bo, bm=self.Bm)
                case "bf":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf + d,
                                              bi=self.Bi, bo=self.Bo, bm=self.Bm)
                case "bi":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf,
                                              bi=self.Bi + d, bo=self.Bo, bm=self.Bm)
                case "bo":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf, bi=self.Bi,
                                              bo=self.Bo + d, bm=self.Bm)
                case "bm":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf, bi=self.Bi,
                                              bo=self.Bo, bm=self.Bm + d)
                case "default":
                    return self.forward_check(wf=self.Wf, wi=self.Wi, wo=self.Wo, wm=self.Wm, bf=self.Bf, bi=self.Bi,
                                              bo=self.Bo, bm=self.Bm)

        dx = 1e-7
        if self.output_size != 0:
            for i, j, k in zip(self.output_check(self.Wy + dx, self.By).values(), self.output_check(self.Wy, self.By + dx).values(),
                               self.output_check(self.Wy, self.By).values()):
                self.cWy += (i - k) / dx
                self.cBy += (j - k) / dx

        for t, j in h(d=dx).items():
            self.cWf += (h("wf", d=dx)[t] - j) / dx
            self.cWi += (h("wi", d=dx)[t] - j) / dx
            self.cWo += (h("wo", d=dx)[t] - j) / dx
            self.cWm += (h("wm", d=dx)[t] - j) / dx
            self.cBf += (h("bf", d=dx)[t] - j) / dx
            self.cBi += (h("bi", d=dx)[t] - j) / dx
            self.cBo += (h("bo", d=dx)[t] - j) / dx
            self.cBm += (h("bm", d=dx)[t] - j) / dx

        dcwf = np.average(np.absolute(self.cWf - self.dWf))
        dcwi = np.average(np.absolute(self.cWi - self.dWi))
        dcwo = np.average(np.absolute(self.cWo - self.dWo))
        dcwm = np.average(np.absolute(self.cWm - self.dWm))
        dcbf = np.average(np.absolute(self.cBf - self.dBf))
        dcbi = np.average(np.absolute(self.cBi - self.dBi))
        dcbo = np.average(np.absolute(self.cBo - self.dBo))
        dcbm = np.average(np.absolute(self.cBm - self.dBm))
        print(f"Layer: {self.number}, Wf-error: {dcwf:.2e}")
        print(f"Layer: {self.number}, Wi-error: {dcwi:.2e}")
        print(f"Layer: {self.number}, Wo-error: {dcwo:.2e}")
        print(f"Layer: {self.number}, Wm-error: {dcwm:.2e}")
        print(f"Layer: {self.number}, Bf-error: {dcbf:.2e}")
        print(f"Layer: {self.number}, Bi-error: {dcbi:.2e}")
        print(f"Layer: {self.number}, Bo-error: {dcbo:.2e}")
        print(f"Layer: {self.number}, Bm-error: {dcbm:.2e}")
        sum_diff = dcwf + dcwi + dcwo + dcwm + dcbf + dcbi + dcbo + dcbm
        if self.output_size != 0:
            print(f"Output Wy-error: {np.average(np.absolute(self.cWy - self.dWy)):.2e}")
            print(f"Output By-error: {np.average(np.absolute(self.cBy - self.dBy)):.2e}")
        return sum_diff

    def sample__input(self, x0: int) -> np.ndarray:
        x = np.zeros((self.input_size, 1))
        x[x0] = 1
        return x

    def sample_lstm(self, x: np.ndarray, h_prev: np.ndarray, c: np.ndarray) -> (np.ndarray, np.ndarray):
        x = np.concatenate((x, h_prev), axis=0)
        f = self.sigmoid(self.Wf @ x + self.Bf)
        i = self.sigmoid(self.Wi @ x + self.Bi)
        o = np.tanh(self.Wo @ x + self.Bo)
        m = self.sigmoid(self.Wm @ x + self.Bm)
        c = f * c + i * m
        h = o * c
        return h, c

    def sample_out(self, h: np.ndarray) -> int:
        y = self.Wy @ h + self.By
        y = y - np.max(y)
        p = (np.exp(y) / np.sum(np.exp(y))).ravel()
        try:
            ix = int(np.random.choice(range(self.output_size), size=1, p=p))
        except ValueError as e:
            print(e)
            print(p)
            ix = 0
        return ix

    def change_output(self, output_size: int) -> None:
        self.output_size = output_size
        self.Wy = np.random.randn(self.output_size, self.hidden_size)
        self.By = np.random.randn(self.output_size, 1)


class LSTM:
    def __init__(self, layer_sizes: list[int], sequence_length: int, learning_rate: float, training_data: str):
        # Training data
        self.data: str = training_data
        # Length of a single sequence
        self.sequence_length: int = sequence_length
        # Formatted training data as a list of characters
        self.data_set: list[str] = sorted(list(set(training_data)))
        # The learning rate
        self.learning_rate: float = learning_rate
        # The amount of different characters in the data set
        self.dict_size = len(self.data_set)
        # Converting the characters into integers and vice-versa
        self.char_to_ix = {ch: i for i, ch in enumerate(self.data_set)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.data_set)}
        # The sizes of the different layers
        self.layer_sizes: list[int] = layer_sizes
        # Initialising the previous hidden and cell states
        self.prev_hidde = [np.zeros((i, 1)) for i in self.layer_sizes]
        self.prev_cells = [np.zeros((i, 1)) for i in self.layer_sizes]
        prev_size = self.layer_sizes[0]
        self.layers = []
        # Initialise the layers to a list
        for layer, size in enumerate(self.layer_sizes):
            if layer == 0:
                # First layer parameters: Vocab size and hidden size
                self.layers.append(LSTMLayer(layer, self.dict_size, size, self.learning_rate))
            elif layer == len(layer_sizes) - 1:
                # Last layer previous hidden size and output is the vocab size
                self.layers.append(LSTMLayer(layer, prev_size, size, self.learning_rate, output_size=self.dict_size))
                prev_size = size
            else:
                # Initialising the rest of the layers
                self.layers.append(LSTMLayer(layer, prev_size, size, self.learning_rate))
                prev_size = size

        print(f"Layers of the network with dictionary of {self.dict_size} characters:")
        for i in self.layers:
            print(f"\t{i}")

        self.loss: list[float] = [0]

    def initialise_layers(self) -> None:
        # Initialise the parameters for each layer before an iteration
        for layer in self.layers:
            layer.initialise_layer()

    def train(self):
        # Initialising the counters n the iteration p the sequence counter
        n, p = 0, 0

        while True:
            # Check if p fits inside the data
            if p >= len(self.data):
                p = 0
            # Set the training data for the iteration
            iter_in = [self.char_to_ix[i] for i in self.data[p: p + self.sequence_length]]
            iter_out = [self.char_to_ix[i] for i in self.data[p + 1: p + self.sequence_length + 1]]

            self.initialise_layers()
            out: dict[int, np.ndarray] = {}
            for layerN, layer in enumerate(self.layers):
                if layerN == 0:
                    layer.format_text_input(iter_in)
                    out: dict[int, np.ndarray] = layer.forward_pass()
                elif layerN == len(self.layers) - 1:
                    layer.format_non_text_input(out)
                    out = layer.output_layer()
                    self.loss.append(float(layer.loss(iter_out)))
                elif layerN >= len(self.layers) - 1:
                    continue
                else:
                    layer.format_non_text_input(out)
                    out = layer.forward_pass()

            dh = {}
            # Backpropagation
            for layerN, layer in enumerate(reversed(self.layers)):
                if layerN == 0:
                    dy = layer.backpropagation_output_layer(iter_out)
                    dh = layer.backpropagation_lstm_layer(dy)
                else:
                    dh = layer.backpropagation_lstm_layer(dh)
            if n % 1000 == 0:
                for layer in self.layers:
                    layer.grad_check()
            # Update parameters
            for layer in self.layers:
                layer.sgd()
            # Print out the sample and the details
            if n % 50 == 0:
                print(self.details(n), "\n")
                plt.plot(self.loss)
                plt.savefig("lossLSTM.png")
                plt.close()

            if n % 100 == 0:
                print(f"Sample: \n[{self.sample(100)}]\n")

            if n % 1000 == 0:
                print(self.sample(1000))
            # Increment n and p at the end of the iteration
            n += 1
            p += self.sequence_length

    def sample(self, size):
        # Randomise the first character
        rand_x = np.random.randint(0, self.dict_size)
        x = self.layers[0].sample__input(rand_x)
        chars = [int(rand_x)]
        text = "".join(self.ix_to_char[chars[0]])
        txt = ""
        hiddens = [np.zeros((i.hidden_size, 1)) for i in self.layers]
        cells = [np.zeros((i.hidden_size, 1)) for i in self.layers]
        for n in range(size):
            for i, layer in enumerate(self.layers):
                x = layer.sample_lstm(x, hiddens[i], cells[i])
                hiddens[i] = x[0]
                cells[i] = x[1]
                x = x[0]
            y = self.layers[-1].sample_out(x)
            x = self.layers[0].sample__input(y)
            chars.append(self.ix_to_char[y])
            text += chars[-1]
            txt = ""
            for char in text:
                if char == "\n":
                    txt += "[/linebreak]"
                else:
                    txt += char
        return txt

    def details(self, n):
        print(f"n: {n}, loss: {self.loss[-1]}, timestep sample: [{self.sample(10)}]")
