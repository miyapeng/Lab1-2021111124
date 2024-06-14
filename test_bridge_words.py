import unittest
from lab1 import create_graph, query_bridge_words

class TestQueryBridgeWords(unittest.TestCase):

    def setUp(self):
        # Setting up the graph for each test case
        data1 = "a b c b a"
        self.graph1 = create_graph(data1)

    def test_query_bridge_words_case1(self):
        result = query_bridge_words(self.graph1, "a", "c")
        print(f'Testcase1 real output:{result}')
        self.assertEqual(result, 'The bridge words from"a"to"c"are:b')
        print('Testcase1 exp output: The bridge words from"a"to"c"are:b')
        print('Testcase1 Pass')

    def test_query_bridge_words_case2(self):
        result = query_bridge_words(self.graph1, "a", "b")
        print(f'Testcase2 real output:{result}')
        self.assertEqual(result, 'No bridge words from "a" to "b" !')
        print('Testcase2 exp output: No bridge words from "a" to "b" !')
        print('Testcase2 Pass')


    def test_query_bridge_words_case3(self):
        result = query_bridge_words(self.graph1, "a", "y")
        print(f'Testcase3 real output:{result}')
        self.assertEqual(result, 'No "y" in the graph !')
        print('Testcase3 exp output: No "y" in the graph !')
        print('Testcase3 Pass')


    def test_query_bridge_words_case4(self):
        result = query_bridge_words(self.graph1, "x", "b")
        print(f'Testcase4 real output:{result}')
        self.assertEqual(result, 'No "x" in the graph !')
        print('Testcase4 exp output: No "x" in the graph !')
        print('Testcase4 Pass')

    def test_query_bridge_words_case5(self):
        result = query_bridge_words(self.graph1, "x", "y")
        print(f'Testcase5 real output:{result}')
        self.assertEqual(result, 'No "x" and "y" in the graph !')
        print('Testcase5 exp output: No "x" and "y" in the graph !')
        print('Testcase5 Pass')

if __name__ == '__main__':
    unittest.main()
