import heapq
import time
import random

import gymnasium as gym
import numpy as np

import lbforaging
from lbforaging.foraging import Player, Plan

def astar(grid_size, start, goal):
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
    p = 2
    f = 2
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
    # First observation
    obss, info = env.reset()

    done = False

    # Format (y, x, level) food then (y, x, level) player
    plans = []
    player: Player

    # Initial sampling steps to get potential plans with associated probability and rewards
    for idx, player in enumerate(env.players):
        local_obs = np.reshape(obss[idx],(-1,3))
        food_obss = local_obs[:num_food,:]
        player_loc = np.array(player.position)

        # Plans for each player
        plan = Plan()
        for food_idx in range(num_food):
            food_loc = food_obss[food_idx,:2]
            food_level = food_obss[food_idx,2]
            
            # Simple astar planning
            path = astar((s,s), tuple(player_loc), tuple(food_loc))
            reward = [0]
            total_reward = 0
            for i in range(len(path)-2):
                total_reward -= 1
                reward.append(total_reward)
            
            # First run so all plans have the same probabilitity
            plan.plans.append(path)
            plan.rewards.append(reward)
            plan.probability.append(1.0 / num_food)
        plans.append(plan)

    # Probability optimisation
    for idx, player in enumerate(env.players):
        print("Player: ", idx)
        plan = plans[idx]

        # # Randomly choose one plan
        # plan_indx = list(range(len(plan.plans)))
        # chosen_plan_idx = random.choices(plan_indx, weights=plan.probability, k=1)[0]

        # Optimise for probability using the chosen plan
        for each_plan_indx,_ in enumerate(plan.plans):
            # Calculate the expected reward for each plan
            # which is also the probability times the reward 
            expected_reward = plan.probability[each_plan_indx] * plan.rewards[each_plan_indx][-1]
            print(f"Expected reward for plan {each_plan_indx}: {expected_reward}")
            
            # For each other robot, sample their plans based on the given probability and calculate the expected reward for choosing that plan
            for other_idx, other_player in enumerate(env.players):
                if idx == other_idx:
                    continue
                # Sample other robot plans
                other_plan_indices = list(range(len(plans[other_idx].plans)))
                other_chosen_plan_idx = random.choices(other_plan_indices, weights=plans[other_idx].probability, k=1)[0]
                # print(other_chosen_plan_idx)
                other_expected_reward = plans[other_idx].probability[other_chosen_plan_idx] * plans[other_idx].rewards[other_chosen_plan_idx][-1]
                print(other_expected_reward)

        # option = plan.plans[chosen_plan_idx]
        # option_reward = plan.rewards[chosen_plan_idx]
        # option_probability = plan.probability[chosen_plan_idx]
        # total_reward_for_option = np.sum(option_reward)
        # print(total_reward_for_option)


        # for option_idx,_ in enumerate(plan.plans[1:]):
        #     option = plan.plans[option_idx]
        #     option_reward = plan.rewards[option_idx]
        #     option_probability = plan.probability[option_idx]
            
        #     total_reward_for_option = np.sum(option_reward)
            
        # for other_idx, other_player in enumerate(env.players):
        #     if idx == other_idx:
        #         continue
        #     other_plan = plans[other_idx]
        #     for i in range(len(plan.plans)):
        #         for j in range(len(other_plan.plans)):
        #             if plan.plans[i] == other_plan.plans[j]:
        #                 plan.probability[i] += other_plan.probability[j]
        #                 plan.rewards[i] += other_plan.rewards[j]
        #                 plan.probability[j] = 0
        #                 other_plan.rewards[j] = 0


    env.set_plans(plans)

    while True:
        # actions = env.action_space.sample() 
        # obss, rewards, done, _, _ = env.step((0,0))
        
        env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
