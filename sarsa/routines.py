
def train(agent, env, video_maker=None, episode_number=None):
    ## train for one episode.

    if episode_number:
        print('Episode: {}'.format(episode_number))

    episode_len = 0
    frames = []

    s, _, done, info = env.reset()
    a = agent.act(s)

    frames.append(info['frame'])

    while not done:

        s_prime, r, done, info = env.step(a)
        a_prime = agent.act(s_prime)
        _ = agent.learn(s, a, r, s_prime, a_prime, done)

        s = s_prime
        a = a_prime
        episode_len += 1

        frames.append(info['frame'])

    if video_maker:
        _ = video_maker.save(frames, episode_number=episode_number)

    print('\ttimesteps to goal... {}'.format(episode_len))
    return episode_len


def play(agent, env, video_maker=None):
    ## play through an episode without training
    episode_len = 0
    frames = []

    s, _, done, info = env.reset()
    a = agent.act(s)

    frames.append(info['frame'])

    while not done:

        s_prime, r, done, info = env.step(a)
        a_prime = agent.act(s_prime)

        s = s_prime
        a = a_prime
        episode_len += 1

        frames.append(info['frame'])

    if video_maker is not None:
        fp = video_maker.save(frames)
        print('\nVideo of trained agent playing through an episode:')
        print('\t' + fp)

    return episode_len