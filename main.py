import LSTM as rnn


def main():
	data = open(r"inputs.txt", "r", encoding="utf-8").read()
	network = rnn.LSTM([8, 6], 20, 1e-2, data)
	network.train()

if __name__ == "__main__":
	main()
