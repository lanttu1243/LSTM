import numpy as np
from matplotlib import pyplot as plt


class LSTM_layer:

	def __init__(self, layer_number: int, input_size: int, hidden_size: int, learning_rate: float, output_size: int = 0):
		# Initialising the static parameters for the layer
		self.hidden_size: int = hidden_size  # Hidden size is the number of neurons in a layer
		self.input_size: int = input_size   # Input size is the number of inputs into the layer
		self.learning_rate: float = learning_rate  # Learning rate tells the model how quickly to adjust parameters
		self.smooth_loss: float = 0   # Initialising the loss to zero
		self.number: int = layer_number  # The number of the layer in the network
		self.output_size: int = output_size  # If the layer is the last layer this tells the size of the output often the same as the input for the first layer

		# define the variables used in the layer
		self.xs: dict[int, np.ndarray] = {}    # Input
		self.hs: dict[int, np.ndarray] = {}    # Hidden state
		self.fg: dict[int, np.ndarray] = {}    # Forget gate
		self.ig: dict[int, np.ndarray] = {}    # Forget gate
		self.og: dict[int, np.ndarray] = {}    # Forget gate
		self.mg: dict[int, np.ndarray] = {}    # Forget gate
		self.cs: dict[int, np.ndarray] = {}    # Forget gate
		self.ch: dict[int, np.ndarray] = {}    # Forget gate
		self.ys: dict[int, np.ndarray] = {}    # Output
		self.ps: dict[int, np.ndarray] = {}    # Probabilities
		self.conc: dict[int, np.ndarray] = {}  # the concatenation of xs(t) and hs(t-1)

		# Initialize weights and biases for the lstm unit
		self.Wf: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.001
		self.Wi: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.001
		self.Wo: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.001
		self.Wm: np.ndarray = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * 0.001

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

		self.prev_hidden: np.ndarray = np.zeros((self.hidden_size, 1))
		self.prev_cell: np.ndarray = np.zeros((self.hidden_size, 1))

	def __str__(self):
		# Return the layer as a string
		if self.output_size != 0:
			out_size = self.output_size
		else:
			out_size = self.hidden_size
		return f"LSTM(Layer no. {self.number}, input size: {self.input_size}, hidden size: {self.hidden_size}, output size: {out_size})"

	@staticmethod
	def sigmoid(x: np.ndarray) -> np.ndarray:
		return 1 / (1 + np.exp(- x))

	@staticmethod
	def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
		return x * (1 - x)

	@staticmethod
	def tanh_derivative(x: np.ndarray) -> np.ndarray:
		return 1 - x ** 2

	def initialise_Layer(self) -> None:
		# initialise the parameters for the iteration to empty or zero
		self.xs, self.hs, self.fg, self.ig, self.og, self.mg, self.cs, self.ch = {}, {}, {}, {}, {}, {}, {}, {}
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
			self.ch[t] = np.tanh(self.cs[t])
			self.hs[t] = self.og[t] * self.ch[t]
		return self.hs

	def output_layer(self) -> dict[int, np.ndarray]:
		hid: dict[int, np.ndarray] = self.forward_pass().copy()
		hid.pop(-1)
		for t, k in enumerate(hid):
			self.ys[t] = self.Wy @ self.hs[t-1] + self.By
			self.ps[t] = np.exp(self.ys[t] - np.max(self.ys[t])) / np.sum(np.exp(self.ys[t] - np.max(self.ys[t])))
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

	def backpropagation_LSTM_layer(self, din: dict[int, np.ndarray]) -> dict[int, np.ndarray]:

		dcnext: np.ndarray = np.zeros((self.hidden_size, 1))
		dfnext: np.ndarray = np.zeros((self.hidden_size, 1))
		dinext: np.ndarray = np.zeros((self.hidden_size, 1))
		donext: np.ndarray = np.zeros((self.hidden_size, 1))
		dmnext: np.ndarray = np.zeros((self.hidden_size, 1))

		dout: dict[int, np.ndarray] = {}

		for t in reversed(range(len(din))):
			dinh = din[t] * self.og[t]
			dhc = dinh * self.tanh_derivative(self.ch[t]) + dcnext
			dcnext += dhc * self.fg[t]

			dcf = dhc * self.cs[t - 1] + dfnext
			dfraw = dcf * self.sigmoid_derivative(self.fg[t])
			self.dBf += dfraw
			self.dWf += dfraw @ self.conc[t].T
			dfout = self.Wf[:, :self.input_size].T @ dfraw
			dfnext += self.Wf[:, self.input_size:].T @ dfraw

			dci = dhc * self.mg[t] + dinext
			diraw = dci * self.sigmoid_derivative(self.ig[t])
			self.dBi += diraw
			self.dWi += diraw @ self.conc[t].T
			diout = self.Wi[:, :self.input_size].T @ diraw
			dinext += self.Wi[:, self.input_size:].T @ diraw

			dino = din[t] * self.ch[t] + donext
			doraw = dino * self.sigmoid_derivative(self.og[t])
			self.dBo += doraw
			self.dWo += doraw @ self.conc[t].T
			doout = self.Wo[:, :self.input_size].T @ doraw
			donext += self.Wo[:, self.input_size:].T @ doraw

			dcm = dhc * self.ig[t] + dmnext
			dmraw = dcm * self.tanh_derivative(self.mg[t])
			self.dBm += dmraw
			self.dWm += dmraw @ self.conc[t].T
			dmout = self.Wm[:, :self.input_size].T @ dmraw
			dmnext += self.Wm[:, self.input_size:].T @ dmraw

			dout[t] = dfout + diout + dmout + doout
		for i in [self.dWf, self.dWi, self.dWo, self.dWm, self.dBf, self.dBi, self.dBo, self.dBm, self.dWy, self.dBy]:
			np.clip(i, -1, 1, out=i)
		return dout

	def adagrad(self) -> None:
		self.mWf = np.square(self.dWf)
		self.mWi = np.square(self.dWi)
		self.mWo = np.square(self.dWo)
		self.mWm = np.square(self.dWm)

		self.mBf = np.square(self.dBf)
		self.mBi = np.square(self.dBi)
		self.mBo = np.square(self.dBo)
		self.mBm = np.square(self.dBm)

		self.mWy = np.square(self.dWy)
		self.mBy = np.square(self.dBy)

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

	def SGD(self) -> None:
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
			loss += - np.log(np.clip(self.ps[t][targets[t]], 1e-9, 1 - 1e-9))
		self.smooth_loss = 0.99 * self.smooth_loss + 0.01 * loss
		return self.smooth_loss

	def sample__input(self, x0: int) -> np.ndarray:
		x = np.zeros((self.input_size, 1))
		x[x0] = 1
		return x

	def sample_LSTM(self, x: np.ndarray, h_prev: np.ndarray, c: np.ndarray) -> (np.ndarray, np.ndarray):
		x = np.concatenate((x, h_prev), axis=0)
		f = self.sigmoid(self.Wf @ x + self.Bf)
		i = self.sigmoid(self.Wi @ x + self.Bi)
		o = np.tanh(self.Wo @ x + self.Bo)
		m = self.sigmoid(self.Wm @ x + self.Bm)
		c = f * c + i * m
		h = o * np.tanh(c)
		return h, c

	def sample_out(self, h: np.ndarray) -> int:
		y = self.Wy @ h + self.By
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
				self.layers.append(LSTM_layer(layer, self.dict_size, size, self.learning_rate))
			elif layer == len(layer_sizes) - 1:
				# Last layer previous hidden size and output is the vocab size
				self.layers.append(LSTM_layer(layer, prev_size, size, self.learning_rate, output_size=self.dict_size))
				prev_size = size
			else:
				# Initialising the rest of the layers
				self.layers.append(LSTM_layer(layer, prev_size, size, self.learning_rate))
				prev_size = size

		print(f"Layers of the network with dictionary of {self.dict_size} characters:")
		for i in self.layers:
			print(f"\t{i}")

		self.loss: list[float] = [0]

	def initialise_layers(self) -> None:
		# Initialise the parameters for each layer before an iteration
		for layer in self.layers:
			layer.initialise_Layer()

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
					dh = layer.backpropagation_LSTM_layer(dy)
				else:
					dh = layer.backpropagation_LSTM_layer(dh)
			# Update parameters
			for layer in self.layers:
				layer.adagrad()
			# Print out the sample and the details
			if n % 500 == 0:
				print(self.details(n), "\n")
				plt.plot(self.loss)
				plt.savefig("lossLSTM.png")
				plt.close()

			if n % 1000 == 0:
				print(f"Sample: \n[{self.sample(100)}]\n")
			# Increment n and p at the end of the iteration
			n += 1
			p += self.sequence_length

	def sample(self, size):
		# Randomise the first character
		rand_x = np.random.randint(0, self.dict_size)
		x = self.layers[0].sample__input(rand_x)
		chars = [int(rand_x)]
		text = "".join(self.ix_to_char[chars[0]])
		hiddens = [np.zeros((i.hidden_size, 1)) for i in self.layers]
		cells = [np.zeros((i.hidden_size, 1)) for i in self.layers]
		for n in range(size):
			for i, layer in enumerate(self.layers):
				x = layer.sample_LSTM(x, hiddens[i], cells[i])
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
		return text

	def details(self, n):
		print(f"n: {n}, loss: {self.loss[-1]}, timestep sample: [{self.sample(30)}]")
