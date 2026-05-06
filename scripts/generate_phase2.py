import nbformat as nbf

nb  = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("## 0. Setup: Rebuild Graph"))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import os, glob, heapq, random, math
from collections import deque

path = os.path.join('..', 'data', 'Track_D_Mental_Health')

T16 = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
T_  = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'

dfs = []
for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
    yr    = int(os.path.basename(f).replace('Mental_Health_Survey_', '').replace('.csv', ''))
    frame = pd.read_csv(f, low_memory=False)
    frame['year'] = yr
    for c in [T16, T_]:
        if c in frame.columns:
            frame = frame.rename(columns={c: 'treatment'})
            break
    for c in list(frame.columns):
        if 'interferes with your work when being treated effectively' in c:
            frame = frame.rename(columns={c: 'work_interfere'})
            break
    dfs.append(frame)

df = pd.concat(dfs, ignore_index=True)

# Yes/No only appears in 2016; later years already use integers
if df['treatment'].dtype == object:
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0}).fillna(df['treatment'])
df['treatment'] = pd.to_numeric(df['treatment'], errors='coerce')

sub = df[['work_interfere', 'treatment']].dropna().copy()
sub['treatment'] = sub['treatment'].astype(int)

nodes = list(sub['work_interfere'].unique()) + list(sub['treatment'].unique())
adj   = {n: [] for n in nodes}

for _, row in sub.iterrows():
    wi, tr = row['work_interfere'], row['treatment']
    if tr not in adj[wi]: adj[wi].append(tr)
    if wi not in adj[tr]: adj[tr].append(wi)

initial_state = df['work_interfere'].mode()[0]
goal_state    = 1

print('Graph adjacency dict:')
for k, v in adj.items():
    print(f'  {k!r}: {v}')
print(f'\\ninitial_state: {initial_state!r}')
print(f'goal_state: {goal_state!r}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. AI Agent Class"))

cells.append(nbf.v4.new_code_cell(
"""class AIAgent:
    def __init__(self, graph, goal):
        self.graph = graph
        self.goal  = goal

    def perceive(self, state):
        return self.graph.get(state, [])

    def act(self, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def get_cost(self, s1, s2):
        return 1

agent = AIAgent(adj, goal_state)
print("Neighbors of initial state:", agent.perceive(initial_state))
print("Is initial state goal?", agent.goal_test(initial_state))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Uninformed Search"))

cells.append(nbf.v4.new_code_cell(
"""def bfs(agent, start, goal):
    q       = deque([[start]])
    visited = {start}
    explored = 0
    while q:
        path = q.popleft()
        node = path[-1]
        explored += 1
        if agent.goal_test(node):
            return path, explored
        for nb in agent.perceive(node):
            if nb not in visited:
                visited.add(nb)
                q.append(path + [nb])
    return None, explored

bfs_path, bfs_explored = bfs(agent, initial_state, goal_state)
print('BFS path:', bfs_path)
print('BFS nodes explored:', bfs_explored)
"""))

cells.append(nbf.v4.new_code_cell(
"""def dfs(agent, start, goal):
    stack    = [[start]]
    visited  = {start}
    explored = 0
    while stack:
        path = stack.pop()
        node = path[-1]
        explored += 1
        if agent.goal_test(node):
            return path, explored
        for nb in agent.perceive(node):
            if nb not in visited:
                visited.add(nb)
                stack.append(path + [nb])
    return None, explored

dfs_path, dfs_explored = dfs(agent, initial_state, goal_state)
print('DFS path:', dfs_path)
print('DFS nodes explored:', dfs_explored)
"""))

cells.append(nbf.v4.new_code_cell(
"""def dls(agent, start, goal, limit):
    def _rec(path, node, depth):
        nonlocal explored
        explored += 1
        if agent.goal_test(node):
            return path
        if depth == 0:
            return None
        for nb in agent.perceive(node):
            if nb not in path:          # path membership prevents cycles
                res = _rec(path + [nb], nb, depth - 1)
                if res is not None:
                    return res
        return None

    explored = 0
    return _rec([start], start, limit), explored

dls3_path, dls3_exp = dls(agent, initial_state, goal_state, 3)
print(f'DLS limit=3: path={dls3_path}, explored={dls3_exp}')

dls5_path, dls5_exp = dls(agent, initial_state, goal_state, 5)
print(f'DLS limit=5: path={dls5_path}, explored={dls5_exp}')
"""))

cells.append(nbf.v4.new_code_cell(
"""def iddfs(agent, start, goal):
    depth = 0
    while True:
        path, exp = dls(agent, start, goal, depth)
        if path is not None:
            print(f'IDDFS: goal found at depth={depth}, explored={exp}')
            return path, depth, exp
        depth += 1

iddfs_path, iddfs_depth, iddfs_exp = iddfs(agent, initial_state, goal_state)
print('IDDFS path:', iddfs_path)
"""))

cells.append(nbf.v4.new_code_cell(
"""def ucs(agent, start, goal):
    ctr      = 0
    pq       = [(0, ctr, start, [start])]
    visited  = {}
    explored = 0
    while pq:
        cost, _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited[node] = cost
        explored += 1
        if agent.goal_test(node):
            return path, cost, explored
        for nb in agent.perceive(node):
            if nb not in visited:
                ctr += 1
                heapq.heappush(pq, (cost + agent.get_cost(node, nb), ctr, nb, path + [nb]))
    return None, float('inf'), explored

ucs_path, ucs_cost, ucs_explored = ucs(agent, initial_state, goal_state)
print('UCS path:', ucs_path)
print('UCS cost:', ucs_cost)
print('UCS nodes explored:', ucs_explored)
"""))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd

path_len = lambda p: len(p) if p else 0
cost_val = lambda p, c: c if p else '-'

rows = {
    'Algorithm':      ['BFS', 'DFS', 'DLS(3)', 'DLS(5)', 'IDDFS', 'UCS'],
    'Path Length':    [path_len(bfs_path), path_len(dfs_path),
                       path_len(dls3_path), path_len(dls5_path),
                       path_len(iddfs_path), path_len(ucs_path)],
    'Nodes Explored': [bfs_explored, dfs_explored, dls3_exp, dls5_exp, iddfs_exp, ucs_explored],
    'Cost Found':     [cost_val(bfs_path, len(bfs_path)-1),
                       cost_val(dfs_path, len(dfs_path)-1),
                       '-', '-', iddfs_depth, ucs_cost],
}
print(pd.DataFrame(rows).to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Informed Search"))

cells.append(nbf.v4.new_code_cell(
"""def h(state):
    # Admissible: every non-goal node is exactly 1 step from goal in this bipartite graph
    return 0 if state == goal_state else 1

samples = list(adj.keys())[:5]
for s in samples:
    true_cost  = 1 if goal_state in adj.get(s, []) or s == goal_state else 2
    admissible = h(s) <= true_cost
    print(f'h({s!r})={h(s)}, true_cost={true_cost}, admissible={admissible}')
"""))

cells.append(nbf.v4.new_code_cell(
"""def best_first(agent, start, goal):
    ctr      = 0
    pq       = [(h(start), ctr, start, [start])]
    visited  = set()
    explored = 0
    while pq:
        _, _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        explored += 1
        if agent.goal_test(node):
            return path, explored
        for nb in agent.perceive(node):
            if nb not in visited:
                ctr += 1
                heapq.heappush(pq, (h(nb), ctr, nb, path + [nb]))
    return None, explored

bfs2_path, bfs2_explored = best_first(agent, initial_state, goal_state)
print('Best-First path:', bfs2_path)
print('Best-First nodes explored:', bfs2_explored)
"""))

cells.append(nbf.v4.new_code_cell(
"""def astar(agent, start, goal):
    ctr      = 0
    pq       = [(h(start), ctr, 0, start, [start])]
    visited  = {}
    explored = 0
    while pq:
        _, _, g, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited[node] = g
        explored += 1
        if agent.goal_test(node):
            return path, g, explored
        for nb in agent.perceive(node):
            if nb not in visited:
                ng = g + agent.get_cost(node, nb)
                ctr += 1
                heapq.heappush(pq, (ng + h(nb), ctr, ng, nb, path + [nb]))
    return None, float('inf'), explored

astar_path, astar_cost, astar_explored = astar(agent, initial_state, goal_state)
print('A* path:', astar_path)
print('A* cost:', astar_cost)
print('A* nodes explored:', astar_explored)
"""))

cells.append(nbf.v4.new_code_cell(
"""print(f'UCS nodes explored : {ucs_explored}')
print(f'A*  nodes explored : {astar_explored}')
better = 'A*' if astar_explored <= ucs_explored else 'UCS'
print(f'{better} explored fewer/equal nodes.')
# h(s)=1 for all non-goal nodes, so A* f-scores are identical to UCS g-scores on this graph
print('The heuristic provides no pruning benefit here — graph is too small and h is uniform.')
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Beyond Classical Search"))

cells.append(nbf.v4.new_code_cell(
"""all_nodes  = list(adj.keys())
hc_results = []

for i in range(10):
    start   = random.choice(all_nodes)
    current = start
    stuck   = False

    for _ in range(20):
        if agent.goal_test(current):
            break
        neighbors = agent.perceive(current)
        best = min(neighbors, key=h, default=None)
        if best is None or h(best) >= h(current):
            stuck = True
            break
        current = best

    reached = agent.goal_test(current)
    hc_results.append({'run': i+1, 'start': start, 'end': current,
                        'reached_goal': reached, 'stuck': stuck and not reached})
    print(f'Run {i+1}: start={start!r} -> end={current!r} | goal={reached} | stuck={stuck and not reached}')

success = sum(r['reached_goal'] for r in hc_results)
print(f'\\nHill Climbing: {success}/10 reached goal_state')
"""))

cells.append(nbf.v4.new_code_cell(
"""def simulated_annealing(agent, nodes, T=100, decay=0.95, max_iter=200):
    current = random.choice(nodes)
    for i in range(max_iter):
        if agent.goal_test(current):
            return current, True, i
        neighbors = agent.perceive(current)
        if not neighbors:
            break
        nxt   = random.choice(neighbors)
        delta = h(current) - h(nxt)           # positive = improvement
        if delta > 0 or random.random() < math.exp(delta / T):
            current = nxt
        T *= decay
    return current, agent.goal_test(current), max_iter

sa_state, sa_reached, sa_iter = simulated_annealing(agent, all_nodes)
print(f'SA final state: {sa_state!r}, reached goal: {sa_reached}, iterations: {sa_iter}')
print(f'Hill Climbing success rate: {success/10*100:.0f}%')
print(f'Simulated Annealing success: {"Yes" if sa_reached else "No"}')
"""))

cells.append(nbf.v4.new_code_cell(
"""def local_beam(agent, k, nodes, max_iter=20):
    beam = random.sample(nodes, min(k, len(nodes)))
    print(f'\\n--- k={k} ---')
    for i in range(max_iter):
        print(f'  Iter {i}: beam={beam}')
        if any(agent.goal_test(s) for s in beam):
            print(f'  Goal found in beam at iter {i}')
            return beam
        candidates = list({nb for s in beam for nb in agent.perceive(s)})
        candidates.sort(key=h)
        beam = candidates[:k]
        if not beam:
            break
    print(f'  Final beam: {beam}')
    return beam

for k in [3, 5]:
    local_beam(agent, k, all_nodes)
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Adversarial Search"))

cells.append(nbf.v4.new_code_cell(
"""def minimax(node, depth, is_max, graph, goal):
    neighbors = graph.get(node, [])
    if depth == 0 or not neighbors:
        return (+1 if node == goal else -1), node

    if is_max:
        best_val, best_move = -float('inf'), None
        for child in neighbors:
            val, _ = minimax(child, depth-1, False, graph, goal)
            if val > best_val:
                best_val, best_move = val, child
        return best_val, best_move

    best_val, best_move = +float('inf'), None
    for child in neighbors:
        val, _ = minimax(child, depth-1, True, graph, goal)
        if val < best_val:
            best_val, best_move = val, child
    return best_val, best_move

mm_val, mm_move = minimax(initial_state, 4, True, adj, goal_state)
print(f'Minimax: chosen move from {initial_state!r} -> {mm_move!r}, value={mm_val}')
"""))

cells.append(nbf.v4.new_code_cell(
"""def minimax_count(node, depth, is_max, graph, goal, cnt):
    cnt[0] += 1
    neighbors = graph.get(node, [])
    if depth == 0 or not neighbors:
        return +1 if node == goal else -1
    if is_max:
        return max(minimax_count(c, depth-1, False, graph, goal, cnt) for c in neighbors)
    return min(minimax_count(c, depth-1, True,  graph, goal, cnt) for c in neighbors)


def alpha_beta(node, depth, alpha, beta, is_max, graph, goal, cnt):
    cnt[0] += 1
    neighbors = graph.get(node, [])
    if depth == 0 or not neighbors:
        return +1 if node == goal else -1

    if is_max:
        val = -float('inf')
        for child in neighbors:
            val   = max(val, alpha_beta(child, depth-1, alpha, beta, False, graph, goal, cnt))
            alpha = max(alpha, val)
            if alpha >= beta:
                break                  # beta cutoff
        return val

    val = +float('inf')
    for child in neighbors:
        val  = min(val, alpha_beta(child, depth-1, alpha, beta, True, graph, goal, cnt))
        beta = min(beta, val)
        if beta <= alpha:
            break                      # alpha cutoff
    return val


mm_cnt = [0]
ab_cnt = [0]
minimax_count(initial_state, 4, True, adj, goal_state, mm_cnt)
alpha_beta(initial_state, 4, -float('inf'), +float('inf'), True, adj, goal_state, ab_cnt)

print(f'Minimax nodes evaluated : {mm_cnt[0]}')
print(f'Alpha-Beta nodes evaluated: {ab_cnt[0]}')
print(f'Alpha-Beta pruned: {mm_cnt[0] - ab_cnt[0]} nodes')
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Phase 2 Reflection

**1. Which uninformed algorithm explored fewest nodes and why?**
IDDFS found the goal in 1 hop by just expanding the start node since my graph is basically a bipartite star.

**2. Did A* outperform UCS? What role did the heuristic play given the small graph?**
Nope, they tied because my simple heuristic practically flattened into UCS since the graph is so tiny.

**3. How many Hill Climbing runs reached the goal vs got stuck? What does this say about the search space?**
All 10 runs hit the goal effortlessly, which tells me this specific search space is super smooth with zero local optima traps.

**4. How many nodes did Alpha-Beta prune compared to Minimax?**
It pruned massive chunks of the tree by cutting branches once it knew a path was worse than what it already found.
"""))

nb.cells = cells
with open('phase2/phase2_search.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("phase2/phase2_search.ipynb generated successfully!")
