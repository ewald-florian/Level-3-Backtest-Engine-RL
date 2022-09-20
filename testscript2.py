from replay.replay import Replay
import numpy as np

re = Replay(identifier="FME", start_date="2022-02-16", end_date="2022-02-16", shuffle=False, frequency="5m", exclude_high_activity_time=True)

re._generate_episode_start_list()

#print(np.array(re.episode_start_list))

re.build_new_episode()

print("Epsisode Msg List len ", len(re.episode.message_packet_list))

print(type(re.episode.episode_start))