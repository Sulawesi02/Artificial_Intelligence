import math
import random
import copy

# 切换玩家颜色
def SwapColor(color):
    if color == 'X':
        return 'O'
    return 'X'

class Node:
    def __init__(self, parent, color, board, action, C=1):
        self.parent = parent
        self.children = []
        self.count = 0  # 节点模拟次数
        self.value = 0  # 节点得分
        self.board = board  # 节点对应的棋盘状态
        self.color = color  # 当前颜色
        self.action = action  # 从父节点到当前节点的落子动作
        self.C = C  # 参数

class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, max_time=60):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.max_time = max_time

    def get_ucb1(self, node, t):
        if node.count == 0:
            return float('INF')
        return node.value / node.count + node.C * ((2 * math.log(t) / node.count) ** 0.5)

    def MCTS(self, root):
        root.board.display()  # 显示当前的棋盘状态。
        for i in range(1, self.max_time):
            select_node = self.select(root, i)  # 选择
            leaf_node = self.expand(select_node)  # 扩展
            value = self.stimulate(leaf_node)  # 模拟
            self.backup(leaf_node, value)  # 反向传播
        max_ucb = -float('INF')
        max_child = None
        for child in root.children:
            ucb1 = self.get_ucb1(child, self.max_time)
            if ucb1 > max_ucb:
                max_ucb = ucb1
                max_child = child
        return max_child.action

    # 选择
    def select(self, node, t):
        if len(node.children) == 0:
            return node
        else:
            max_ucb = -float('INF')
            max_child = None
            for child in node.children:
                ucb1 = self.get_ucb1(child, t)
                if ucb1 > max_ucb:
                    max_ucb = ucb1
                    max_child = child
            return self.select(max_child, t)  # 递归选择 UCB1 值最大的子节点

    # 扩展
    def expand(self, node):
        if node.count == 0:
            return node
        else:
            for action in list(node.board.get_legal_actions(node.color)):
                board = copy.deepcopy(node.board)
                board._move(action, node.color)
                node.children.append(Node(node, SwapColor(node.color), board, action))
            if len(node.children) == 0:
                return node
            return random.choice(node.children)

    # 模拟
    def stimulate(self, node):
        board = copy.deepcopy(node.board)
        while not self.game_over(board):
            legal_actions = list(board.get_legal_actions(node.color))
            if len(legal_actions) > 0:
                action = random.choice(legal_actions)
                board._move(action, node.color)
                node.color = SwapColor(node.color)
            winner, diff = board.get_winner()
            if winner == 2:  # 平局
                return 0
            elif (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):  # AIPlayer 获胜
                return 1
            else:  # AIPlayer 失败
                return 0

    # 反向传播
    def backup(self, node, value):
        while node is not None:
            node.count += 1
            node.value += value
            node = node.parent

    def get_move(self, board):
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        root = Node(None, self.color, board, None)
        action = self.MCTS(root)
        return action

    def game_over(self, board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over = len(b_list) == 0 and len(w_list) == 0
        return is_over