import Node
import training
import testing_node
import testing_leaf
import prepare_dataset


def node(test_node, test_leaf):
    node_dataset, left_dataset, right_dataset, test_dataset = prepare_dataset.prepare_dataset([[1], [7]])
    PATHS = ['./node_net.pth', './left_net.pth', './right_net.pth']

    if test_node:
        for h in range(10):
            training.train(left_dataset, PATHS[1])
            training.train(right_dataset, PATHS[2])
            for i in range(10, 60, 10):
                print(f'Number of epochs: {i}')
                node = Node.Node(node_dataset, PATHS[1], PATHS[2], PATHS[0])
                node.train(i)
                testing_node.test(test_dataset, PATHS)
            print('NEW TRY')

    if test_leaf:
        training.train(left_dataset, PATHS[1])
        training.train(right_dataset, PATHS[2])
        testing_leaf.test(test_dataset, PATHS[1])
        testing_leaf.test(test_dataset, PATHS[2])


if __name__ == '__main__':
    node(False, True)


