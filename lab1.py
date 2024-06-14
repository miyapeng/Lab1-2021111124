# pylint: disable=E
'''lab1'''
import re
import heapq
import random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QLineEdit, QTextEdit, QPushButton,
                             QFileDialog)


def process_file(filename):
    '''加载文件'''
    with open(filename, 'r', encoding="utf-8") as file:
        data = file.read().replace('\n', ' ')
    return data


def create_graph(data):
    '''根据文件建立图结构'''
    data = re.sub(r'[^a-zA-Z\s]', ' ', data)
    words = data.lower().split()
    graph_raw = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 1):
        graph_raw[words[i]][words[i + 1]] += 1
    return graph_raw


def query_bridge_words(graph, word1, word2):
    '''查询桥接词'''
    graph_node = []
    for node in graph:
        graph_node.append(node)
    for node in graph:
        for neighbor, _ in graph[node].items():
            if neighbor not in graph_node:
                graph_node.append(neighbor)
    if word1 not in graph_node and word2 not in graph_node:
        return f"No \"{word1}\" and \"{word2}\" in the graph !"
    if word1 not in graph_node:
        return f"No \"{word1}\" in the graph !"
    if word2 not in graph_node:
        return f"No \"{word2}\" in the graph !"
    bridge_words = [word for word in graph[word1] if word2 in graph[word]]
    if len(bridge_words) == 0:
        return f"No bridge words from \"{word1}\" to \"{word2}\" !"
    bridge_words = ', '.join(bridge_words)
    return f"The bridge words from\"{word1}\"to\"{word2}\"are:{bridge_words}"


def query_bridge_words_2(graph, word1, word2):
    '''查询桥接词'''
    graph_node = []
    for node in graph:
        graph_node.append(node)
    for node in graph:
        for neighbor, _ in graph[node].items():
            if neighbor not in graph_node:
                graph_node.append(neighbor)
    if word1 not in graph_node:
        return []
    if word2 not in graph_node:
        return []
    bridge_words = [word for word in graph[word1] if word2 in graph[word]]
    if len(bridge_words) == 0:
        return []
    return bridge_words


def generate_new_text(graph, text):
    '''生成桥接文本'''
    words = text.split()
    new_words = [words[0]]
    for i in range(len(words) - 1):
        bridge_words = query_bridge_words_2(graph, words[i], words[i + 1])
        if (len(bridge_words) > 0):
            new_words.append(random.choice(bridge_words))
        new_words.append(words[i + 1])
    return ' '.join(new_words)


def show_directed_graph(graph_raw):
    '''展示所生成的图'''
    graph = nx.DiGraph()

    for word1 in graph_raw:
        for word2 in graph_raw[word1]:
            graph.add_edge(word1, word2, weight=graph_raw[word1][word2])

    pos = nx.kamada_kawai_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(graph,
                     pos,
                     with_labels=True,
                     node_color='skyblue',
                     edge_color='r',
                     node_size=170,
                     font_size=8)
    nx.draw_networkx_edge_labels(graph,
                                 pos,
                                 edge_labels=nx.get_edge_attributes(
                                     graph, 'weight'))
    plt.savefig("graph.png", dpi=300)
    plt.show()  # 添加这行代码来显示图像


def show_directed_graph_addfor_git(graph_raw):
    '''显示图像'''
    graph = nx.DiGraph()

    for word1 in graph_raw:
        for word2 in graph_raw[word1]:
            graph.add_edge(word1, word2, weight=graph_raw[word1][word2])

    pos = nx.kamada_kawai_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(graph,
                     pos,
                     with_labels=True,
                     node_color='skyblue',
                     edge_color='r',
                     node_size=170,
                     font_size=8)
    nx.draw_networkx_edge_labels(graph,
                                 pos,
                                 edge_labels=nx.get_edge_attributes(
                                     graph, 'weight'))
    plt.savefig("graph.png", dpi=300)
    plt.show()  # 添加这行代码来显示图像


def dijkstra(graph, start, end=None):
    '''最短路迪杰斯特拉算法'''
    distances = defaultdict(lambda: 1e10, {node: 1e10 for node in graph})
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances.get(current_node, 1e10):
            continue

        for neighbor, weight in graph.get(current_node, {}).items():
            distance = current_distance + weight

            if distance < distances.get(neighbor, 1e10):
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    if end:
        path = []
        node = end
        if distances.get(end, 1e10) >= (1e10) / 2:
            return (1e10, [])
        while previous_nodes.get(node, None) is not None:
            path.append(node)
            node = previous_nodes[node]
        path.append(start)
        path.reverse()
        return (distances[end], path)

    return distances, previous_nodes


# Function to find the shortest path between two nodes
def calculate_shortest_path(graph, start_node, end_node):
    graph_node = []
    for node in graph:
        graph_node.append(node)
    for node in graph:
        for neighbor, _ in graph[node].items():
            if neighbor not in graph_node:
                graph_node.append(neighbor)
    if start_node not in graph_node:
        if end_node not in graph_node:
            s = "start node and end node not in graph!"
            return s
        else:
            s = "start node not in graph!"
            return s
    else:
        if end_node not in graph_node:
            s = "end node not in graph!"
            return s
        else:
            distance, path = dijkstra(graph, start_node, end_node)
            if distance == 1e10:
                s = "No way!"
                return s
            else:
                s = f"Shortest paths from {start_node} to {end_node}:{path} \
                       with distance {distance}"
                display_paths(graph, path, start_node, end_node)
                return s


def display_paths(graph_raw, path, start_node, end_node):
    '''展示最短路'''
    graph = nx.DiGraph()
    nodes = []
    for node in graph_raw:
        nodes.append(node)
    for node in graph_raw:
        for neighbor, weight in graph_raw[node].items():
            if neighbor not in nodes:
                nodes.append(neighbor)
    for node in nodes:
        for neighbor, weight in graph_raw.get(node, {}).items():
            graph.add_edge(node, neighbor, weight=weight)

    pos = nx.kamada_kawai_layout(graph)
    edges = graph.edges(data=True)
    plt.figure()
    nx.draw(graph,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            font_size=10)
    nx.draw_networkx_edge_labels(graph,
                                 pos,
                                 edge_labels={
                                     (u, v): d['weight']
                                     for u, v, d in edges
                                 })

    if path:
        for i in range(len(path) - 1):
            nx.draw_networkx_edges(graph,
                                   pos,
                                   edgelist=[(path[i], path[i + 1])],
                                   width=2.5,
                                   alpha=0.6,
                                   edge_color=["red"])
        # Highlight the start and end nodes
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=[start_node],
                               node_color='green',
                               node_size=700)
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=[end_node],
                               node_color='red',
                               node_size=700)

    plt.title(f"Shortest Path from {start_node} to {end_node}")
    plt.show()


def calculate_shortest_path_all(graph, start):
    '''计算所有的最短路'''
    distances, previous_nodes = dijkstra(graph, start)
    all_paths = {}
    nodes = []
    for node in graph:
        nodes.append(node)
    for node in graph:
        for neighbor, _ in graph[node].items():
            if neighbor not in nodes:
                nodes.append(neighbor)
    for node in nodes:
        if node == start:
            continue
        path = []
        current = node
        if distances.get(node, 1e10) == 1e10:
            all_paths[node] = "No way"
            continue
        while previous_nodes.get(current, None) is not None:
            path.append(current)
            current = previous_nodes[current]
        path.append(start)
        path.reverse()
        all_paths[node] = path

    return all_paths


class App(QWidget):
    '''界面展示'''
    def __init__(self):
        '''初始化界面'''
        super().__init__()
        self.title = 'Shortest Path Finder'
        self.init_ui()
        self.stop_requested = False

    def init_ui(self):
        '''UI界面定义各种功能'''
        self.setWindowTitle(self.title)

        main_layout = QHBoxLayout()  # Use QHBoxLayout for main layout
        up_layout = QVBoxLayout()
        left_layout = QVBoxLayout()  # Left side layout
        middle_layout = QVBoxLayout()  # Middle side layout
        right_layout = QVBoxLayout()  # Right side layout

        # File path input widgets
        self.filepath_label = QLabel('Enter file path:', self)
        up_layout.addWidget(self.filepath_label)

        self.filepath_input = QLineEdit(self)
        up_layout.addWidget(self.filepath_input)

        self.browse_button = QPushButton('Browse', self)
        self.browse_button.clicked.connect(self.browse_file)
        up_layout.addWidget(self.browse_button)
        self.save_picture_button = QPushButton('Save Picture', self)
        self.save_picture_button.clicked.connect(self.save_picture)
        up_layout.addWidget(self.save_picture_button)
        # Left side widgets
        self.input_label = QLabel('Enter number of nodes (1 or 2):', self)
        left_layout.addWidget(self.input_label)

        self.node_count_input = QLineEdit(self)
        left_layout.addWidget(self.node_count_input)

        self.start_node_label = QLabel('Enter start node:', self)
        left_layout.addWidget(self.start_node_label)

        self.start_node_input = QLineEdit(self)
        left_layout.addWidget(self.start_node_input)

        self.end_node_label = QLabel('Enter end node (if applicable):', self)
        left_layout.addWidget(self.end_node_label)

        self.end_node_input = QLineEdit(self)
        left_layout.addWidget(self.end_node_input)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        left_layout.addWidget(self.result_text)

        self.run_button = QPushButton('Find Path', self)
        self.run_button.clicked.connect(self.on_click)
        left_layout.addWidget(self.run_button)

        self.walk_button = QPushButton('Random Walk', self)
        self.walk_button.clicked.connect(self.random_walk)
        left_layout.addWidget(self.walk_button)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_walk)
        left_layout.addWidget(self.stop_button)

        # Middle side widgets
        self.bridge_source_label = QLabel('Enter bridge source word:', self)
        middle_layout.addWidget(self.bridge_source_label)

        self.bridge_source_input = QLineEdit(self)
        middle_layout.addWidget(self.bridge_source_input)

        self.bridge_end_label = QLabel('Enter bridge end word:', self)
        middle_layout.addWidget(self.bridge_end_label)

        self.bridge_end_input = QLineEdit(self)
        middle_layout.addWidget(self.bridge_end_input)

        self.bridge_result_text = QTextEdit(self)
        self.bridge_result_text.setReadOnly(True)
        middle_layout.addWidget(self.bridge_result_text)

        self.bridge_button = QPushButton('Find Bridge Words', self)
        self.bridge_button.clicked.connect(self.on_bridge_click)
        middle_layout.addWidget(self.bridge_button)

        # Right side widgets
        self.bridge_in_text_label = QLabel(
            'Enter text for bridge word insertion:', self)
        right_layout.addWidget(self.bridge_in_text_label)

        self.bridge_in_text = QLineEdit(self)
        right_layout.addWidget(self.bridge_in_text)

        self.bridge_out_text_label = QLabel('New text with bridge words:',
                                            self)
        right_layout.addWidget(self.bridge_out_text_label)

        self.bridge_out_text = QTextEdit(self)
        self.bridge_out_text.setReadOnly(True)
        right_layout.addWidget(self.bridge_out_text)

        self.generate_button = QPushButton('Generate New Text', self)
        self.generate_button.clicked.connect(self.on_generate_click)
        right_layout.addWidget(self.generate_button)

        # Combine layouts
        main_layout.addLayout(up_layout)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def browse_file(self):
        '''检索文件'''
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options)
        if file_path:
            self.filepath_input.setText(file_path)

    def on_click(self):
        '''点击执行操作'''
        file_path = self.filepath_input.text().strip()
        node_count = int(self.node_count_input.text())
        start_node = self.start_node_input.text().strip()
        end_node = self.end_node_input.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        if node_count == 2:
            s = calculate_shortest_path(graph, start_node, end_node)
            self.result_text.setText(s)
        elif node_count == 1:
            all_paths = calculate_shortest_path_all(graph, start_node)
            result = []
            for end, paths in all_paths.items():
                if paths == "No way":
                    result.append(f"No way to reach {end} from {start_node}")
                else:
                    result.append(f"Paths from {start_node} to {end}: {paths}")
                    display_paths(graph, paths, start_node, end)
            self.result_text.setText("\n".join(result))
            calculate_shortest_path_all(graph, start_node)
        else:
            self.result_text.setText("Invalid input. Please enter 1 or 2.")

    def random_walk(self):
        '''随机游走'''
        self.stop_requested = False
        file_path = self.filepath_input.text().strip()
        data = process_file(file_path)
        graph = create_graph(data)

        start_node = random.choice(list(graph.keys()))
        visited_edges = set()
        path = []
        current_node = start_node
        path.append(current_node)
        while not self.stop_requested:
            neighbors = list(graph[current_node].items())
            if not neighbors:
                break
            next_node, _ = random.choice(neighbors)
            edge = (current_node, next_node)

            if edge in visited_edges:
                path.append(next_node)
                self.stop_requested = True  # 出现重复边后停止
            else:
                visited_edges.add(edge)
                current_node = next_node
                path.append(current_node)

        self.result_text.setText(f"Random walk path: {path}")

        with open("result.txt", "w", encoding="utf-8") as file:
            file.write(" -> ".join(path))

    def stop_walk(self):
        '''停止随机游走'''
        self.stop_requested = True

    def on_bridge_click(self):
        '''桥接词按键'''
        file_path = self.filepath_input.text().strip()
        word1 = self.bridge_source_input.text().strip()
        word2 = self.bridge_end_input.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        result = query_bridge_words(graph, word1, word2)
        self.bridge_result_text.setText(result)

    def on_generate_click(self):
        '''生成桥接文本'''
        file_path = self.filepath_input.text().strip()
        text1 = self.bridge_in_text.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        new_text = generate_new_text(graph, text1)
        self.bridge_out_text.setText(new_text)

    def save_picture(self):
        '''保存图片'''
        file_path = self.filepath_input.text().strip()
        data = process_file(file_path)
        graph_raw = create_graph(data)
        show_directed_graph(graph_raw)


if __name__ == '__main__':
    app = QApplication([])
    window = App()
    window.show()
    app.exec_()
