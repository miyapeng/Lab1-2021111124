import sys
import re
import heapq
import random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit

def process_file(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')
    return data

def create_graph(data):
    data = re.sub(r'[^a-zA-Z\s]', ' ', data)
    words = data.lower().split()
    graph = defaultdict(lambda: defaultdict(int))
    print(graph)
    for i in range(len(words) - 1):
        graph[words[i]][words[i + 1]] += 1
    return graph

def queryBridgeWords(graph, word1, word2):
    if word1 not in graph and word2 not in graph:
        return f"No \"{word1}\" and \"{word2}\" in the graph !"
    if word1 not in graph:
        return f"No \"{word1}\" in the graph !"
    if word2 not in graph:
        return f"No \"{word2}\" in the graph !"
    bridge_words = [word for word in graph[word1] if word2 in graph[word]]
    if len(bridge_words) == 0:
        return f"No bridge words from \"{word1}\" to \"{word2}\" !"
    bridge_words_str = ', '.join(bridge_words)
    return f"The bridge words from \"{word1}\" to \"{word2}\"are: {bridge_words_str}"

def queryBridgeWords_2(graph, word1, word2):
    if word1 not in graph:
        return []
    if word2 not in graph:
        return []
    bridge_words = [word for word in graph[word1] if word2 in graph[word]]
    if len(bridge_words) == 0:
        return []
    return bridge_words

def generateNewText(graph, text):
    words = text.split()
    new_words = [words[0]]
    for i in range(len(words) - 1):
        bridge_words = queryBridgeWords_2(graph, words[i], words[i + 1])
        if (len(bridge_words)>0):
            new_words.append(random.choice(bridge_words))
        new_words.append(words[i + 1])
    return ' '.join(new_words)

def showDirectedGraph(graph):
    G = nx.DiGraph()

    for word1 in graph:
        for word2 in graph[word1]:
            G.add_edge(word1, word2, weight=graph[word1][word2])

    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(12,8))
    nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue', edge_color='r', node_size=170, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.savefig("graph.png",dpi=300)
    plt.show()  # 添加这行代码来显示图像



def dijkstra(graph, start, end=None):
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
        if distances.get(end, 1e10) >= (1e10)/2:
            return (1e10, [])
        while previous_nodes.get(node, None) is not None:
            path.append(node)
            node = previous_nodes[node]
        path.append(start)
        path.reverse()
        return (distances[end], path)
    
    return distances, previous_nodes

# Function to find the shortest path between two nodes
def calcShortestPath(graph, start, end):
    if start not in graph or end not in graph:
        return -1, []
    distance, path = dijkstra(graph, start, end)
    if distance == 1e10:
        return -1, []
    return distance, path
def display_paths(graph, path, start_node, end_node):
    G = nx.DiGraph()
    nodes = []
    for node in graph:
        nodes.append(node)
    for node in graph:
        for neighbor, weight in graph[node].items():
            if neighbor not in nodes:
                 nodes.append(neighbor)
    print(nodes)
    for node in nodes:
        for neighbor, weight in graph.get(node, {}).items():
            G.add_edge(node, neighbor, weight=weight)
    
    pos = nx.kamada_kawai_layout(G)
    edges = G.edges(data=True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(path)-1))
    
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})
    
    if path:
        for i in range(len(path)-1):
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1])], width=2.5, alpha=0.6, edge_color=["red"])
        # Highlight the start and end nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color='red', node_size=700)
    
    plt.title(f"Shortest Path from {start_node} to {end_node}")
    plt.show()



def calcShortestPath_all(graph, start):
    distances, previous_nodes = dijkstra(graph, start)
    all_paths = {}
    nodes=[]
    for node in graph:
        nodes.append(node)
    for node in graph:
        for neighbor, weight in graph[node].items():
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

# Function to display all paths from a start node to all other nodes
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QPushButton, QFileDialog
)
import random
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Shortest Path Finder'
        self.initUI()
        self.stop_requested = False

    def initUI(self):
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
        self.walk_button.clicked.connect(self.randomWalk)
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
        self.bridge_in_text_label = QLabel('Enter text for bridge word insertion:', self)
        right_layout.addWidget(self.bridge_in_text_label)

        self.bridge_in_text = QLineEdit(self)
        right_layout.addWidget(self.bridge_in_text)

        self.bridge_out_text_label = QLabel('New text with bridge words:', self)
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
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            self.filepath_input.setText(file_path)

    def on_click(self):
        file_path = self.filepath_input.text().strip()
        node_count = int(self.node_count_input.text())
        start_node = self.start_node_input.text().strip()
        end_node = self.end_node_input.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        if node_count == 2:
            distance, paths = calcShortestPath(graph, start_node, end_node)
            if distance == -1:
                self.result_text.setText("No way!")
            else:
                self.result_text.setText(f"Shortest paths from {start_node} to {end_node}: {paths} with distance {distance}")
                display_paths(graph, paths, start_node, end_node)
        elif node_count == 1:
            all_paths = calcShortestPath_all(graph, start_node)
            result = []
            for end, paths in all_paths.items():
                if paths == "No way":
                    result.append(f"No way to reach {end} from {start_node}")
                else:
                    result.append(f"Paths from {start_node} to {end}: {paths}")
                    display_paths(graph, paths, start_node, end)
            self.result_text.setText("\n".join(result))
            calcShortestPath_all(graph, start_node)
        else:
            self.result_text.setText("Invalid input. Please enter 1 or 2.")

    def randomWalk(self):
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

        with open("result.txt", "w") as file:
            file.write(" -> ".join(path))

    def stop_walk(self):
        self.stop_requested = True


    def on_bridge_click(self):
        file_path = self.filepath_input.text().strip()
        word1 = self.bridge_source_input.text().strip()
        word2 = self.bridge_end_input.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        result = queryBridgeWords(graph, word1, word2)
        self.bridge_result_text.setText(result)

    def on_generate_click(self):
        file_path = self.filepath_input.text().strip()
        text1 = self.bridge_in_text.text().strip()

        data = process_file(file_path)
        graph = create_graph(data)

        new_text = generateNewText(graph, text1)
        self.bridge_out_text.setText(new_text)
    def save_picture(self):
        file_path = self.filepath_input.text().strip()
        data = process_file(file_path)
        graph = create_graph(data)
        showDirectedGraph(graph)

if __name__ == '__main__':
    app = QApplication([])
    window = App()
    window.show()
    app.exec_()
