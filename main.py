import Node
import training
import testing
import prepare_dataset


def node():
    for h in range(10):
        node_dataset, left_dataset, right_dataset, test_dataset = prepare_dataset.prepare_dataset([[0], [1]])
        PATHS = ['./node_net.pth', './left_net.pth', './right_net.pth']
        training.train(left_dataset, PATHS[1])
        training.train(right_dataset, PATHS[2])
        for i in range(10, 60, 10):
            print(f'Number of epochs: {i}')
            node = Node.Node(node_dataset, PATHS[1], PATHS[2], PATHS[0])
            node.train(i)
            testing.test(test_dataset, PATHS)
        print('NEW TRY')


if __name__ == '__main__':
    node()


