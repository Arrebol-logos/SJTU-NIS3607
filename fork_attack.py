# 导入所需的库
import random
import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List

# 设置随机种子、模拟轮数、节点数量和最大查询次数
TRYS = 10
SEED = 123
ROUNDS = 1000
NODE_NUM = 500
MAX_ORACLE_QUERY = 100
FORK_LENGTH = 6
block_gen_rate = 1e-3
TIMES = 1

# 定义节点类，每个节点都有一个区块链
class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.blockchain = [Block(-1, None, False)]

    def proof_of_work(self, block_gen_rate, max_oracle_query):
        attempt_count = 0
        while attempt_count < max_oracle_query:
            if random.random() < block_gen_rate:
                self.blockchain.append(Block(self.node_id, self.blockchain[-1], False))
                break
            attempt_count += 1
            
    def reset_blockchain(self):
        self.blockchain = [Block(-1, None, False)]

class HonestNode(Node):
    def __init__(self, node_id: int):
        super().__init__(node_id)

class MaliciousNode(HonestNode):
    def __init__(self, node_id: int, malicious_node_count: int):
        super().__init__(node_id)
        self.count = malicious_node_count

    def proof_of_work(self, block_gen_rate, max_oracle_query):
        queries = 0
        while queries < max_oracle_query * self.count:
            if random.random() < block_gen_rate:
                self.blockchain.append(Block(self.node_id, self.blockchain[-1], True))
                break
            queries += 1

# 定义区块类，每个区块都有一个创建者和一个指向前一个区块的指针
class Block:
    def __init__(self, creator_id: int, prev_block, malicious: bool):
        self.creator_id = creator_id
        self.prev_block = prev_block
        self.malicious = malicious

# 定义函数，找出最长的区块链
def find_longest_chain(nodes: List[Node]):
    max_length = max(len(node.blockchain) for node in nodes)
    longest_chain_ids = [node.node_id for node in nodes if len(node.blockchain) == max_length]
    return [random.choice(longest_chain_ids)]

# 定义函数，更新所有节点的区块链为最长的区块链
def update_all_chains(nodes: List[Node], longest_chain_ids):
    max_length = len(nodes[longest_chain_ids[0]].blockchain)
    for node in nodes:
        selected_id = random.choice(longest_chain_ids)
        node.blockchain = copy.deepcopy(nodes[selected_id].blockchain[:max_length])
    return max_length

def is_fork_attack_successful(nodes: List[Node]):
    longest_chain_ids = find_longest_chain(nodes)
    for longest_chain_id in longest_chain_ids:
        longest_chain = nodes[longest_chain_id].blockchain
        # 检查最后一个区块是否为恶意区块
        if longest_chain[-1].malicious:
            return True
    return False

# 定义函数，设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 主函数
if __name__ == "__main__":
    set_random_seed(SEED)

    # 对不同的恶意节点比例进行模拟
    for malicious_ratio in [0.1, 0.2, 0.3, 0.4]:
        print(f"Starting simulation with malicious node ratio: {malicious_ratio}")
        attack_success_count = 0

        # 创建节点，其中一部分是恶意节点
        nodes = [HonestNode(node_id) if node_id < NODE_NUM * (1 - malicious_ratio) else MaliciousNode(node_id, TIMES) for node_id in range(NODE_NUM)]

        progress_bar = tqdm(range(ROUNDS))
        for round_num in progress_bar:
            # 重置所有节点的区块链
            for node in nodes:
                node.reset_blockchain()

            # 进行FORK_LENGTH轮模拟
            for i in range(FORK_LENGTH+1):
                for node in nodes:
                    node.proof_of_work(block_gen_rate, MAX_ORACLE_QUERY)
                longest_chain_ids = find_longest_chain(nodes)
                length = update_all_chains(nodes, longest_chain_ids)
                progress_bar.set_description(f"Max Valid Chain Length: {length}, Chain Growth Rate: {(length - 1) / (round_num + 1)}")

            # 检查分叉攻击是否成功
            if is_fork_attack_successful(nodes):
                attack_success_count += 1

        print(f"Malicious Node Ratio: {malicious_ratio}, Fork Attack Success Probability: {attack_success_count / ROUNDS}")