import heapq
import time

import gymnasium as gym
import numpy as np

import lbforaging
from lbforaging.foraging import Player

def plan(grid_size, start, goal):
    rows, cols = grid_size

    def heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-way connectivity
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                yield (nx, ny)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g = current_g + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                came_from[neighbor] = current

    return None  # Shouldn't happen in fully connected grid

def main():
    s = 10
    p = 1
    f = 1
    c = 1
    num_food = 2
    id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(
        s, p, f, "-coop" if c else "")
    gym.register(
        id=id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 1,
            "field_size": (s, s),
            "sight": s,
            "max_episode_steps": 100,
            "force_coop": c,
            "min_player_level": 1,
            "min_food_level": 1,
            "max_food_level": 1,
            "max_num_food": num_food,
        },
    )

    env = gym.make(id)

    grid = np.zeros((s,s), dtype=int)
    print(grid)
    # First observation
    obss, info = env.reset()

    done = False

    # Format (y, x, level) food then (y, x, level) player
    player: Player
    for idx, player in enumerate(env.players):
        local_obs = np.reshape(obss[idx],(-1,3))
        food_obss = local_obs[:num_food,:]
        player_loc = np.array(player.position)


        for food_idx in range(num_food):
            food_loc = food_obss[food_idx,:2]
            food_level = food_obss[food_idx,2]
            
            path = plan((s,s), tuple(player_loc), tuple(food_loc))
            print(path)

    while True:
        # actions = env.action_space.sample() 
        # obss, rewards, done, _, _ = env.step((0,0))
        
        env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
