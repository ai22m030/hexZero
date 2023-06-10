from mcts import HexModel, self_play, train

from torch.optim import Adam

if __name__ == '__main__':
    model = HexModel(board_size=7)
    games = self_play(model, 100, 100, 0.01)

    learning_rate = 0.001  # You can experiment with this value
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Passing model's parameters to the optimizer

    epochs = 10  # You can set this to the number of epochs you want to train
    model = train(model, games, optimizer, epochs)
