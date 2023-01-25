import LSTM as rnn


def main():
	data = open(r"input.txt", "r", encoding="utf-8").read()
	network = rnn.LSTM([256, 1024, 128], 100, 1e-3, data)
	network.train()

if __name__ == "__main__":
	main()
