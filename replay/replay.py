#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
""" Replay Module for the Level-3 backtest engine """

import random
import datetime
import logging
logging.getLogger().setLevel(logging.INFO)
# !pip install pandas-market-calendars # -> dependency!
import pandas_market_calendars as mcal
import pandas as pd
from episode.episode import Episode
from market.market import Market
from market.market_trade import MarketTrade
from market.market_metrics import MarketMetrics
from context.context import Context
from agent.agent_trade import AgentTrade
from agent.agent_order import OrderManagementSystem
from reinforcement_learning.transition.env_transition \
    import EnvironmentTransition
from reinforcement_learning.action_space.action_storage import ActionStorage


class Replay:

    """
    Replay class generates a list of episode then loads these
    episodes subsequently and provides them for the backtest.
    """

    def __init__(self,
                 rl_agent: "instance of an rl agent" = None,
                 identifier_list: list = None,
                 identifier: str = "BAY",#"BMW",
                 start_date: str = "2021-01-01",#"2022-02-01",
                 end_date: str = "2021-04-30",#"2022-02-02",
                 episode_length: str = "10s",
                 frequency: str = "5m",
                 seed: int = None,
                 shuffle: bool = True,
                 random_identifier: bool = False,
                 exclude_high_activity_time: bool = False,
                 mode: str = "random_episodes",
                 verbose: bool = True,
                 *args,
                 **kwargs
                 ):

        """
        Initialization of Replay is a crucial step when setting up
        a backtest since it is the central class of the backtest.
        :param rl_agent
            class, rl agent to use for RL backtest
        :param identifier_list
            list, list of several identifiers of assets used for backtest
        :param identifier
            str, single identifier if backtest uses only one asset
        :param start_date
            str, start date of time period considered by backtest
        :param end_date
            str, end date of time period considered by backtest
        :param episode_length
            str, length of a single episode
        :param frequency
            str, frequency to split time period into episodes
        :param seed
            int, random seed
        :param shuffle
            bool, shuffle episodes if set True
        :param random_identifier
            bool, if True, asset is selected randomly
        :param exclude_high_activity_time
            bool, if True, high activity times are excluded from backtest
        :param mode
            str, random episode mode
        :param verbose
            bool, if True, infos are printed
        """

        # -- static attributes
        self.identifier_list = identifier_list
        self.identifier = identifier
        self.random_identifier = random_identifier
        self.start_date = start_date
        self.end_date = end_date
        self.episode_length = episode_length
        self.frequency = frequency
        self.seed = seed
        self.shuffle = shuffle
        self.exclude_high_activity_time = exclude_high_activity_time
        self.mode = mode
        self.verbose = verbose

        # -- dynamic attributes
        self.episode = None
        self.episode_start_list = []
        self.episode_counter = 0
        self.episode_index = 0
        self.step_counter = 0
        self.done = False
        self.second_last_step_flag = False

        # -- generate new episode_start_list
        self._generate_episode_start_list()

        # -- rl agent

        # Note:If the replay instance is going to be used for rl,
        # the instantiated rl agent object has to be passed as input arg.
        # This way, I don't need to import different agent in replay.

        if rl_agent:
            self.rl_agent = rl_agent
        # for non-RL applications
        else:
            self.rl_agent = None

    # rl-step . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Note: new rl_step 2022-10-08
    def rl_step(self):
        """
        RL-step method, to be called in the environment.step() method.
        ---------------------------------------------------------------
        Central Method to step the backtesting engine externally based
        on the episode __next__() method. Can be called from an external
        iterative training loop, for example in the "step" method of an
        RL-environment class in Open-AI Gym convention.

        Done and info are stored to EnvironmentTransition.transition which
        can be accessed by Environment. Observation and Reward are received
        in the agent.step and stores to AgentTransition.transition which can
        be accessed by Environment as well.

        Note:
        method does not have own returns, obs and rew are
        returned via AgentTransition, done info is returned via
        EnvironmentTransition.
        """
        # -- market step
        self._market_step()
        # -- save context (blocked liquidity)
        state_l3_blocked = Market.instances['ID'].state_l3_blocked_liquidity
        Context(state_l3_blocked)

        # -- store done, info to env transition
        done = self.done
        info = {}  # TODO: fill info dict
        EnvironmentTransition(done, info)

        # -- rl_agent
        self.rl_agent.step()
        # -- update step_counter
        self.step_counter += 1

    # non-RL step . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def normal_step(self):
        """
        External Stepping:
        -----------------
        Central Method to step the backtesting engine externally based
        on the episode __next__() method.

        To be used for standard backtest applications without reinforcement
        learning.
        """

        # -- market step
        self._market_step()
        # -- step_counter
        self.step_counter += 1

        # store state to context
        state_l3 = Market.instances['ID'].state_l3
        Context(state_l3)

    def _market_step(self):
        """
        Helper method to conduct the crucial market update operations:
            1. check if episode is done
            2. Receive new message packet from episode.__next__
            3. Pass new message packet to Market
        """

        # -- check if episode is done
        if self.episode._step >= (self.episode.__len__() - 1):
            self.done = True

        # -- update market with new message packet from episode
        message_packet = self.episode.__next__()

        # -- pass nex message packet to Market
        Market.instances['ID'].update_simulation_with_exchange_message(
            message_packet)

    # episode management . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _generate_episode_start_list(self):
        """
        Generates a new episode start list which is a list of timestamps that
        can potentially be used as episode starting points.

        Episode_start_list only contains timestamps which are in XETRA
        trading hours and account for the episode length such that the time
        between episode_start and mid- or close auction is always adequate to
        run an entire episode.

        High activity trading times defined as 08:00-08:15 UTC and 16:15-16:30
        UTC can be optionally excluded using the exclude_high_activity_time
        argument.

        The episode list can optionally be shuffled using the shuffle argument,
        it is possible to define a specific seed using the seed argumen.
        """

        # Xetra trading calendar (trading hours, weekends, holidays)
        xetr = mcal.get_calendar('XETR')
        schedule = xetr.schedule(start_date=self.start_date,
                                 end_date=self.end_date)

        # create date range with respective frequency
        self.episode_start_list = list(
            mcal.date_range(schedule, frequency=self.frequency))

        # -- account for market close and mid auction buffers
        # market_close (16:30 UTC)
        market_close = '01/01/10 16:30:00'  # arbitrary date
        # account for market close
        market_close = datetime.datetime.strptime(market_close,
                                                  '%m/%d/%y %H:%M:%S')

        # account for episode_length
        market_close_buffer = market_close - pd.Timedelta(self.episode_length)
        # mid auction (12.00 UTC)

        mid_auction = '01/01/10 12:00:00'  # arbitrary date

        mid_auction = datetime.datetime.strptime(mid_auction,
                                                 '%m/%d/%y %H:%M:%S')
        # account for episode_length
        mid_auction_buffer = mid_auction - pd.Timedelta(self.episode_length)

        # filter for close buffer
        self.episode_start_list = list(
            filter(lambda ts: (ts.time() < market_close_buffer.time()),
                   self.episode_start_list))

        # filter for mid auction buffer
        self.episode_start_list = list(
            filter(lambda ts: (ts.time() >= mid_auction.time() or
                               ts.time() <= mid_auction_buffer.time()),
                   self.episode_start_list))

        # -- exclude high trading activity time (optional)
        if self.exclude_high_activity_time:
            # high trading activity time in the beginning of the trading day
            high_activity_open = '01/01/10 08:15:00'  # arbitrary date
            # account for market close
            high_activity_open = datetime.datetime.strptime(
                high_activity_open, '%m/%d/%y %H:%M:%S')

            # high trading activity time in the end of the trading day
            high_activity_close = '01/01/10 16:15:00'  # arbitrary date
            # account for market close
            high_activity_close = datetime.datetime.strptime(
                high_activity_close, '%m/%d/%y %H:%M:%S')

            # filter
            self.episode_start_list = list(
                filter(lambda ts: (
                        high_activity_open.time() <= ts.time()
                        <= high_activity_close.time()),
                       self.episode_start_list))

        # -- shuffle episode (optional)
        if self.seed:
            random.seed(self.seed)

        if self.shuffle:
            random.shuffle(self.episode_start_list)

        # -- reset episode_counter to 0
        self.episode_counter = 0
        self.episode_index = 0

    def build_new_episode(self):
        """
        Use the next start timestamp in episode_start_list to build the next
        episode. It is not necessarily the case that each start_point in
        episode_start_list is equipped with the necessary data files, hence
        the construction of the new episode might fail. In this case, the
        episode is build with the next start timestamp until an episode can be
        build successfully.

        episiode_counter counts the successfully built episodes.
        episode_index counts the number of trials to build episodes
        and is therefore used to index the episode_start_list.

        A new episode is directly stored in the self.episode attribute.
        """

        # -- update episode parameters
        episode_start = self.episode_start_list[self.episode_index]
        episode_end = episode_start + pd.Timedelta(self.episode_length)

        if self.random_identifier and self.identifier_list:
            identifier = random.choice(self.identifier_list)
        else:
            identifier = self.identifier

        # -- build new episode
        for attempt in range(100):

            try:
                self.episode = Episode(episode_start=episode_start,
                                       episode_end=episode_end,
                                       identifier=identifier,
                                       verbose=self.verbose)

                # update episode_counter/index
                self.episode_counter += 1
                self.episode_index += 1

                if self.verbose:
                    print("(INFO) new episode was build in attempt: {}".format(
                        attempt + 1))

            # return if episode could not be generated
            except:
                if self.verbose:
                    print("(INFO) could not build episode from start_point")
                    # update episode_counter

                self.episode_index += 1

                # update episode parameters
                episode_start = self.episode_start_list[self.episode_index]
                episode_end = episode_start + pd.Timedelta(self.episode_length)
                continue

            break

    # internally stepped replay . . . . . . . . . . . . . . . . . .

    def run_backtest(self):
        """"
        Internal Stepping:
        -----------------
        Central method to step the backtesting library internally. Based
        on episode __iter__ method.
        """
        pass

    # reinforcement learning reset . . . . . . . . . . . . . . . . . . . . . .

    def rl_reset(self):
        """
        Rl_reset is an extension of base_reset which returns the first
        observation. This is mandatory for OpenAI gym custom environments.
        The method is only relevant for RL-based back-testing. For non-RL
        back-testing use base_reset.
        :return first_observation
            np.array, first observation of the episode
        """

        # set done flag to false
        self.done = False
        # Set second last step to False
        self.second_last_step_flag = False
        # -- base resets resets all relevant backtest-engine classes
        self.base_reset()
        # -- reset ActionStorage
        ActionStorage.reset()
        # -- reset RL-Agent
        # Note: MarketInterface, Reward, ObservationSpace are reset
        # inside agent.reset()
        if self.rl_agent:
            self.rl_agent.reset()
        # -- Reset AgentContext
        # AgentContext.reset()
        # -- get first observation (context needs to include initial state)
        first_obs = self.rl_agent.observation_space.holistic_observation()

        return first_obs

        # Further Possible but not necessary resets (RL extension):
        # ------------------------------------------
        # ObservationSpace
        # RlAgent
        # AgentFeatures
        # ActionSpace (currently not used)
        # ActionStorage (currently not used)
        # AgentContext.reset() careful! has to be reset BEFORE the agent

    # base reset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _reset_market(self, snapshot_start):
        """
        Reset the Market by resetting old instances, creating a new instance,
        and restoring the initial market state from a snapshot. The snapshot
        can be either already parsed (Option 1: this is the case for
        start_snapshots generated by build_new_episode) or non-parsed
        (Option 2: raw snapshots from database). Market reset has to be called
        in the beginning of each episode.

        :param snapshot_start:
            dict, can either be parsed or non-parsed snapshot
        """
        # reset old market instance
        Market.reset_instances()
        # create new instance of MarketState
        _ = Market(market_id="ID")

        # -- initialize state from snapshot:

        # Option 1) from parsed snapshot (episode delivers parsed snapshots)
        # note: parsed snapshots have "template_id" instead of "Template_ID"
        some_price = list(snapshot_start[1].keys())[0]
        if "template_id" in snapshot_start[1][some_price][0]:
            Market.instances['ID'].initialize_state_from_parsed_snapshot(
                snapshot=snapshot_start)

        # Option 2) from un-parsed snapshot (raw snapshot from database)
        else:
            Market.instances['ID'].initialize_state(snapshot=snapshot_start)

    def _reset_episode(self):
        """
        Reset episode by creating a new episode. Each new episode is
        counted by the episode_counter.
        """
        self.build_new_episode()

    def base_reset(self):
        """
        Base_reset can be used to reset the backtest engine before a new
        episode when using external stepping for a non-RL application. For RL
        applications use rl_reset() which includes base_reset() automatically.

        Reset to be called before a new episode starts. This includes to build
        the next episode, reset the Market and to store the first market
        state to Context-context_list. Moreover, all class instances are reset,
        including Context.history, MarketTrade.history, OMS.order_list and
        AgentTrade.history.
        The order is important: 1. reset episode, 2. reset context, 3. reset
        market and store to context.

        General Note:
        -------------
        Replay is used as the entry point of externally stepped back-test
        loops to the backtest-library. Therefore, replay contains a universal
        base_reset method which does not reset the replay class but all
        backtest-engine classes when a new episodes starts.

        Rl-Applications:
        ----------------
        In RL-backtests, the RL-environment calls replay.rl_reset() (which in
        turn calls base_reset() to reset the environment and receive the first
        observation and replay.rl_step() to step the environment and receive
        the latest observation, reward,
        done and info.
        """
        # -- build new episode
        self._reset_episode()
        # -- context
        Context.reset_context_list()
        # -- create new Market as independent class attribute
        self._reset_market(snapshot_start=self.episode.snapshot_start)
        # -- store initial state_l3 to context (to generate first observation)
        state_l3 = Market.instances['ID'].state_l3
        Context(state_l3)
        # -- market trade
        MarketTrade.reset_history()
        # -- market metrics
        _ = MarketMetrics()
        # -- Order Management System
        OrderManagementSystem.reset_history()
        # -- AgentTrade
        AgentTrade.reset_history()

        # TODO: check regularly if further resets are necessary
        # Further Possible but not necessary resets (general library):
        # ------------------------------------------
        # MarketMetrics
        # MarketInterface
        # AgentMetrics
        # AgentFeatures (if dynamic attributes!)
        # MarketFeatures (if dynamic attributes!)
