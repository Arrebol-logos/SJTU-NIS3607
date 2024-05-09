import random
from tqdm import tqdm
from typing import List
import copy
import numpy as np
import os
import torch

# 设置随机种子、模拟轮数、节点数量和最大查询次数
RANDOM_SEED = 123
SIMULATION_ROUNDS = 500
NUMBER_OF_NODES = 500
MAX_BLOCK_GENERATION_ATTEMPTS = 100

# 定义节点类
class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.blockchain = [Block(-1, None)]  # 初始区块链包含一个创世区块

    # 挖矿函数，尝试生成新的区块
    def mine_block(self, block_generation_probability, max_attempts):
        attempts = 0
        while attempts < max_attempts:
            if random.random() < block_generation_probability:
                self.blockchain.append(Block(self.node_id, self.blockchain[-1]))
                break
            attempts += 1

# 定义区块类
class Block:
    def __init__(self, creator_id: int, prev_block):
        self.creator_id = creator_id
        self.prev_block = prev_block

# 选择最长链，并将所有节点的区块链更新为最长链
def select_longest_chain(node_list: List[Node]):
    longest_chain_ids = []
    max_chain_length = 0
    for node in node_list:
        if len(node.blockchain) > max_chain_length:
            max_chain_length = len(node.blockchain)
    for node in node_list:
        if len(node.blockchain) == max_chain_length:
            longest_chain_ids.append(node.node_id)

    for node in node_list:
        selected_id = random.choice(longest_chain_ids)
        node.blockchain = copy.deepcopy(node_list[selected_id].blockchain[:max_chain_length])
    return max_chain_length

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 主函数
if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)

    for block_generation_probability in [1e-7, 1e-6, 1e-5, 1e-4]:
        number_of_honest_nodes = NUMBER_OF_NODES
        nodes = [Node(node_id) for node_id in range(number_of_honest_nodes)]
        print(f"Block Generation Probability: {block_generation_probability}")
        chain_length_per_round = []

        progress_bar = tqdm(range(SIMULATION_ROUNDS))
        for round_num in progress_bar:
            for node in nodes:
                node.mine_block(block_generation_probability, MAX_BLOCK_GENERATION_ATTEMPTS)
            length = select_longest_chain(nodes)
            chain_length_per_round.append(length)
            progress_bar.set_description(f"Max Valid Chain Length: {length}, Chain Growth Rate: {(length - 1) / (round_num + 1)}")

        chain_length_per_round = np.array(chain_length_per_round)
        print(f"Block Generation Probability: {block_generation_probability}, Chain Growth Rate: {(length - 1) / SIMULATION_ROUNDS}")