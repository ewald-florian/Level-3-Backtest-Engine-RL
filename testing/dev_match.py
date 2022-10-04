# Development of new match method which includes SMP



# 1) Build Simulation State without agent orders
#-------------------------------------------------------------------------

simulation_state = {}
# check if active (99999) agent messages with active timestamp exist

# todo: this condition is still helpful
if list(filter(lambda d: d['template_id'] == 99999 and
                         d['timestamp'] <= self.timestamp,
               OMS.order_list)):

    # dicts to create simulation_state
    bid_side_dict = {}
    ask_side_dict = {}
    # lists for values to calculate thresholds
    # (lists needed if agent messages exist only on one side)
    bid_threshold_values = []
    ask_threshold_values = []

    # -- define thresholds for relevant state
    # (i.e. which price levels need to be present)

    # filter prices from agent messages
    buy_prices = []
    sell_prices = []

    for message in OMS.order_list:

        # filter for order submissions (exclude cancellations):
        # Note: account for LATENCY via timestamp
        if message['template_id'] == 99999 and message[
                            'timestamp'] <= self.timestamp:
            if message['side'] == 1:
                buy_prices.append(message['price'])
            if message['side'] == 2:
                sell_prices.append(message['price'])

    # compute min/max buy/sell prices
    if len(buy_prices) > 0:
        max_buy_order_price = max(buy_prices)
        bid_threshold_values.append(max_buy_order_price)
        min_buy_order_price = min(buy_prices)
        ask_threshold_values.append(min_buy_order_price)
    if len(sell_prices) > 0:
        max_sell_order_price = max(sell_prices)
        ask_threshold_values.append(max_sell_order_price)
        min_sell_order_price = min(sell_prices)
        bid_threshold_values.append(min_sell_order_price)

    # compute thresholds:
    ask_threshold = max(ask_threshold_values)
    bid_threshold = min(bid_threshold_values)

    # Bid-side-keys
    bid_keys = list(self._state[1].keys())
    if bid_keys:
        bid_keys_relevant = list(
            i for i in bid_keys if i >= bid_threshold)
    # Ask-side-keys
    ask_keys = list(self._state[2].keys())
    ask_keys_relevant = list(i for i in ask_keys if i <= ask_threshold)

    # -- construct simulation_state from self._state

    # store relevant bid levels to bid_side_dict
    # Note: According to my tests, copy is sufficient to not
    # affect internal state
    for key in bid_keys_relevant:
        if key in self._state[1]:
            bid_side_dict[key] = self._state[1][key].copy()

    # store relevant ask levels to ask_side_dict
    for key in ask_keys_relevant:
        if key in self._state[2]:
            ask_side_dict[key] = self._state[2][key].copy()

    # store relevant levels to simulation_state
    simulation_state[1] = bid_side_dict
    simulation_state[2] = ask_side_dict

    # 2) Agent-Exhausted Liquidity (actually part of simulation_state)
    #-------------------------------------------------------------------------
    # block agent exhausted liquidity
    if self.model_market_impact and self.agent_exhausted_liquidity:
        simulation_state = self._block_agent_exhausted_liquidity(
            state_to_match=simulation_state)

    #-------------------------------------------------------------------------


    # 4) Sort Agent Orders by priority
    #-------------------------------------------------------------------------

    # TODO
    agent_orders = list(filter(lambda d: d['template_id'] == 99999
                                         and d['timestamp'] <= self.timestamp
                                         and 'impact_flag' not in d.keys(),
                                        OMS.order_list))

    # # buy orders: descending price, ascending time
    agent_buy_orders = sorted(filter(lambda d: d['side']==1, agent_orders),
                              key=lambda d: (-d['price'], d['timestamp']))
    # sell orders: ascending price ascending time
    agent_sell_orders = sorted(filter(lambda d: d['side']==2, agent_orders),
                               key=lambda d: (d['price'], d['timestamp']))

    # 5.1) Iterate over agent orders and 5.2) match
    #-------------------------------------------------------------------------
    # todo: is it a problem that the order is adhoc? theoretisch könnten agent
    #  orders orders aus dem state entfernen wodurch agent orders auf der
    #  anderen seite eine höhere priorität haben

    # empty trade_list before all agent orders are matched
    trade_list = []

    for order_list in [agent_buy_orders, agent_sell_orders]:

        # append agent order to state_to_match
        for order in order_list:
            price = order['price']
            side = order['side']
            #todo: name of state?
            if price in state_to_match[side]:
                state_to_match[side]['price'].append(order.copy())
            else:
                state_to_match[side]['price'] = [order.copy()]

        # 5.2 MATCH
        # --------------------------------------------------------------------
        if state_to_match[1] and state_to_match[2]:

            # loop until best price levels cannot be matched
            while True:

                # break if not both sides are filled with orders
                if not state_to_match[1] or not state_to_match[2]:
                    break

                max_buy = max(state_to_match[1].keys())  # best bid
                min_sell = min(state_to_match[2].keys())  # best ask

                # break if order book not crossed
                if not max_buy >= min_sell:
                    break

                # if empty, remove the current best price_level and continue
                if not (state_to_match[1][max_buy]):
                    del state_to_match[1][max_buy]
                    continue

                if not (state_to_match[2][min_sell]):
                    del state_to_match[2][min_sell]
                    continue

                # sort by priority time and select first order (smallest
                # priority timestamp), necessary for simulation!
                order_buy = sorted(state_to_match[1][max_buy],
                                   key=lambda d: d['timestamp'])[0]
                order_sell = sorted(state_to_match[2][min_sell],
                                    key=lambda d: d['timestamp'])[0]

                # TODO: in edge cases the more aggressive price is deciding,
                #  not the timestamp
                # aggressor order has later timestamp
                order_standing, order_aggressor = sorted(
                    [order_buy, order_sell], key=lambda x: x["timestamp"])

                # execution price is always the price of the standing order
                execution_price = order_standing["price"]
                aggressor_side = order_aggressor["side"]
                aggressor_timestamp = order_aggressor["timestamp"]

                if order_buy['quantity'] < order_sell['quantity']:
                    # remove qt from partially executed sell order
                    order_sell["quantity"] -= order_buy["quantity"]
                    # save executed quantity for trade report
                    execution_quantity = order_buy["quantity"]
                    # remove the fully executed buy order from state
                    state_to_match[1][max_buy].remove(order_buy)

                    # mark order as executed (only relevant for impact orders)
                    if "message_id" in order_buy.keys():
                        order_buy["template_id"] = 11111

                elif order_buy['quantity'] > order_sell['quantity']:
                    # remove qt from partially executed buy order
                    order_buy["quantity"] -= order_sell["quantity"]
                    # store to trade-list
                    execution_quantity = order_sell["quantity"]
                    # remove executed sell order from simulation state
                    state_to_match[2][min_sell].remove(order_sell)

                    # mark order as executed (only affects impact orders)
                    if "message_id" in order_sell.keys():
                        order_sell["template_id"] = 11111

                else:
                    # both orders fully executed
                    execution_quantity = order_sell["quantity"]
                    # remove both fully executed orders from simulation state
                    state_to_match[1][max_buy].remove(order_buy)
                    state_to_match[2][min_sell].remove(order_sell)

                    # mark order as executed (only affects impact orders)
                    if "message_id" in order_buy.keys():
                        order_buy["template_id"] = 11111
                    if "message_id" in order_sell.keys():
                        order_sell["template_id"] = 11111

                # append "execution summary" to trade list
                match_execution_summary = {
                    "aggressor_side": aggressor_side,
                    "price": execution_price,
                    "timestamp": aggressor_timestamp,
                    "quantity": execution_quantity,
                    }

                # if agent-message was matched, add message_id
                if "message_id" in order_buy.keys():
                    match_execution_summary["message_id"] = order_buy[
                        "message_id"]
                    match_execution_summary["agent_side"] = 1
                elif "message_id" in order_sell.keys():
                    match_execution_summary["message_id"] = order_sell[
                        "message_id"]
                    match_execution_summary["agent_side"] = 2

                # todo: this shouldnt be possible anymore so remove
                # Edge-Case, both sides are agent orders:
                elif "message_id" in order_buy.keys() and "message_id" in \
                        order_sell.keys():
                    match_execution_summary["message_id"] = order_sell[
                        ["message_id"],
                        order_buy["message_id"]]
                else:
                    pass

                #### DEVELOPMENT ###
                # add lob order to exec summary (for market impact modeling)
                #  if one order is an agent order and the other order is not
                if ("message_id" not in order_sell.keys()
                        and "message_id" in order_buy.keys()):
                    match_execution_summary["lob_order"] = order_sell
                elif ("message_id" not in order_buy.keys()
                      and "message_id" in order_sell.keys()):
                    match_execution_summary["lob_order"] = order_buy
                ############

                trade_list.append(match_execution_summary)

        else:# if not both sides exist
            pass

    #TODO: DIE AGENT ORDERS MÜSSEN NACH DEM MATCHING ENTFERNT WERDEN, SONST
    # KANN ES WIEDER ZU SM KOMMEN!

    # 6) post trade processes
    # ------------------------
    if trade_list:
        self._process_executed_agent_orders(trade_list)
        self._store_agent_trades(trade_list)
        self._process_agent_exhausted_liquidity(trade_list)



