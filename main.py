import Node
import training
import prepare_dataset


def node():
    dataset, left_dataset, right_dataset = prepare_dataset.prepare_dataset([[5], [7]])
    PATHS = ['./node_net.pth', './left_net.pth', './right_net.pth']
    training.train(left_dataset, PATHS[1])
    training.train(right_dataset, PATHS[2])
    node = Node.Node(dataset, PATHS[1], PATHS[2], PATHS[0])
    node.train()


if __name__ == '__main__':
    node()


