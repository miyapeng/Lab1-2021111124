import unittest
from lab1 import create_graph, calculate_shortest_path  # 确保你正确导入了模块和函数

class TestCalculateShortestPath(unittest.TestCase):
    def setUp(self):
        # 用数据创建图
        data1 = "To @ explore strange new worlds,To seek out new life and new civilizations?"
        self.graph1 = create_graph(data1)

    def test_case1(self):
        result = calculate_shortest_path(self.graph1, "x", "y")
        print(f'Output:{result}')
        self.assertEqual(result, 'start node and end node not in graph!')
        print('Testcase1 Pass')

    def test_case2(self):
        result = calculate_shortest_path(self.graph1, "x", "new")
        print(f'Output:{result}')
        self.assertEqual(result, 'start node not in graph!')
        print('Testcase2 Pass')

    def test_case3(self):
        result = calculate_shortest_path(self.graph1, "out", "y")
        print(f'Output:{result}')
        self.assertEqual(result, 'end node not in graph!')
        print('Testcase3 Pass')

    def test_case4(self):
        result = calculate_shortest_path(self.graph1, "civilizations", "to")
        print(f'Output:{result}')
        self.assertEqual(result, 'No way!')
        print('Testcase4 Pass')

    def test_case5(self):
        result = calculate_shortest_path(self.graph1, "new", "to")
        print(f'Output:{result}')
        self.assertEqual(result, "Shortest paths from new to to:['new', 'worlds', 'to']                        with distance 2")
        print('Testcase5 Pass')

if __name__ == '__main__':
    unittest.main()
