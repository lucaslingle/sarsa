from sarsa.agent import SARSA
from sarsa.windy_gridworld import WindyGridworld
from sarsa.routines import train, play
from utils.video_maker import VideoMaker
from utils.plot_maker import PlotMaker
import argparse
import os
import errno

def main(args):
    env = WindyGridworld()
    num_actions = env.num_actions
    num_states = env.num_states

    agent = SARSA(
        num_actions=num_actions, num_states=num_states)

    video_maker = VideoMaker(args.output_dir) if args.save_video else None
    plot_maker = PlotMaker(args.output_dir)

    num_episodes = 200
    total_timesteps = 0
    progress = [(0,0)]

    for episode_number in range(1, num_episodes+1):

        episode_len = train(agent, env, video_maker, episode_number)
        total_timesteps += episode_len
        progress.append((total_timesteps, episode_number))

    _ = play(agent, env, video_maker)
    _ = plot_maker.save(progress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run SARSA on Windy Gridworld.')
    parser.add_argument('--output_dir', type=str, default='output', help='a place to save visualizations')
    parser.add_argument('--save_video', type=bool, default=False, help='save video footage?')
    args = parser.parse_args()

    try:
        os.mkdir(args.output_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    main(args)

