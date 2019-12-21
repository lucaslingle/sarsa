import numpy as np

class WindyGridworld:
    def __init__(self, grid_height=7, grid_width=10, start_location=(3,0), goal_location=(3,7)):

        ## define the state space
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_states = self.grid_height * self.grid_width
        self.state_id_to_location = lambda s: (s // grid_width, s % grid_width)
        self.location_to_state_id = lambda i,j: i * grid_width + j

        ## define the action space
        self.num_actions = 4
        self.action_id_to_direction = {
            0: (-1,0),
            1: (0,1),
            2: (1,0),
            3: (0,-1)
        }

        ## define the environment dynamics - deterministic environment with wind.
        # wind is applied to the state the agent was in, and adds a northward offset equal to the strengths given below.
        # wind strengths depend only on the column the agent was in, and not the row.
        self.wind_strengths = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.location_to_wind_direction = lambda i,j: (-self.wind_strengths[j], 0)

        ## misc.
        self.start_location = start_location
        self.goal_location = goal_location
        self.agent_state = self.location_to_state_id(*self.start_location)
        self.done = False

    def reset(self):
        self.agent_state = self.location_to_state_id(*self.start_location)
        self.done = False
        info = {'frame': self.state_to_frame()}
        return self.agent_state, -1.0, self.done, info

    def step(self, a):
        assert not self.done

        agent_location = self.state_id_to_location(self.agent_state)
        action_direction = self.action_id_to_direction[a]
        wind_direction = self.location_to_wind_direction(*agent_location)
        direction = (action_direction[0] + wind_direction[0], action_direction[1] + wind_direction[1])

        agent_location_new = self.apply_direction(agent_location, direction)
        agent_state_new = self.location_to_state_id(*agent_location_new)
        self.agent_state = agent_state_new
        self.done = (agent_location_new == self.goal_location)

        reward = 0.0 if self.done else -1.0
        info = {
            'frame': self.state_to_frame()
        }

        return self.agent_state, reward, self.done, info

    def apply_direction(self, agent_location, direction):
        agent_location_unclipped = (agent_location[0] + direction[0], agent_location[1] + direction[1])

        agent_location = (
            max(0, min(agent_location_unclipped[0], self.grid_height-1)),
            max(0, min(agent_location_unclipped[1], self.grid_width-1))
        )
        return agent_location

    def state_to_frame(self):
        agent_location = self.state_id_to_location(self.agent_state)
        A = np.zeros(dtype=np.float32, shape=(self.grid_height, self.grid_width, 3))

        A[self.goal_location[0], self.goal_location[1], 1] = 1.0  # green
        A[agent_location[0], agent_location[1], :] = 1.0          # white

        return A




                

