from RNN.LSTM import LSTM

def main():
	data = open(r"C:\Users\lasse\PycharmProjects\RNN\TrainingData.txt", "r", encoding="utf-8").read()
	network = LSTM([16, 32, 16], 5, 1e-4, data)
	network.train()

if __name__ == "__main__":
	main()
