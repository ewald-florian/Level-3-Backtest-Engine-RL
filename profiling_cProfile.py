#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : florian
# Created Date: 07/Sept/2022
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Profile Code to find Bottlenecks:

ncalls : Shows the number of calls made
tottime: Total time taken by the given function. Note that the time made
            in calls to sub-functions are excluded.
percall: Total time / No of calls. ( remainder is left out )
cumtime: Unlike tottime, this includes time spent in this and all
            subfunctions that the higher-level function calls. It is most
            useful and is accurate for recursive functions.

use in terminal:
$ python -m cProfile -s tottime myscript.py
"""
# ---------------------------------------------------------------------------

# TODO: write wrapper function

import cProfile, pstats


def run_code():
    # ----
    from replay.replay import Replay
    replay = Replay()
    replay.reset()
    print("Episode Len: ", replay.episode.__len__())

    for i in range(replay.episode.__len__()):
        replay.rl_step()


def main():
    run_code()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('ncalls')

    stats.print_stats()
    # Remove dir names
    #stats.strip_dirs()
    #stats.print_stats()
    # Export profiler output to file
    #stats.dump_stats('/content/export-data')