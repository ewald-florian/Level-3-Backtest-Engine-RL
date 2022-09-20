from replay.replay import Replay

re = Replay(identifier="FME", start_date="2022-02-16", end_date="2022-02-16")

re._generate_episode_start_list()

re.build_new_episode_new()

print("Epsisode Msg List len ", len(re.episode.episode_message_list))