""" Test and Debug Impact Submissions/Cancellations/Modifications"""


# checklist:
# impact orders in OMS speichern -> funktioniert
# impact orders in den internal state schreiben -> funktioniert
# timestamp plus latency -> funktioniert
# message_id -> funktioniert
# executed impact messages in AgentTrade.history speichern -> funktioniert
# executed impact messages (quantity) aus intrnal-state löschen -> funktioniert
# => d.h. match_new ist für impact orders geeignet!
# impact orders canceln -> funktioniert (template_id ändern und aus internal
# state entfernen.
# impact orders modifizieren -> funktioniert
# template_id von gecancelten orders auf 33333 ändern funktioniert
# template_id wird zu 11111 geändert -> funktioniert
#

# TODO: Problem: wenn impact orders und simulation orders gemeinsam in OMS
#  gespeichert werden, dann werden impact orders auch in den simulations
#  state mathcings ausgeführt und in OMS als executed gekennzeichnet, sie
#  verbeiben jedoch weiterhin (und für immer) im internal state da sie dort
#  nur bei internal-state matching gelöscht werden...
#  Lösungen:
#  1) getrenntes OMS für impact orders
#  2) Einfach vor dem simulation matching impact orders rausfiltern (das ist eigentlich easy,
#  da ich wenn ich den simulation state baue sowieso nach 99999 und timestamp filtere,
#  da kann ich einfach noch zusätzlich nach der impact flag filtern.
#  3) Simualtion und Impact nur getrennt voneinander in einer simulation zulassen
#  entweder odere (es gibt wahrscheinlich eh keinen use case die beiden sachen zu
#  mischen...)

from replay.replay import Replay
from market.market import Market
from context.context import Context
from feature_engineering.market_features import MarketFeatures
from market.market_interface import MarketInterface
from agent.agent_metrics import AgentMetrics
from agent.agent_order import OrderManagementSystem as OMS
from agent.agent_trade import AgentTrade

# Dev MarketFeatures
if __name__ == '__main__':
    replay = Replay(seed=12, shuffle=True)
    replay.base_reset()

    mf = MarketFeatures()
    am = AgentMetrics()
    mi = MarketInterface()

    for i in range(replay.episode.__len__()):

        replay.normal_step()
        # collect context
        Context(Market.instances['ID'].state_l3)
        # generate orders
        if i == 100:
            ticksize = Market.instances['ID'].ticksize
            limit1 = mf.best_ask() + ticksize
            mi.submit_order_impact(side=2, quantity=2220000, limit=limit1)
            print('first order')
            print('OMS: ', OMS.order_list)
            print('AgentTrade', AgentTrade.history)
            print('market')
            if limit1 in Market.instances['ID']._state[2].keys():
                print(Market.instances['ID']._state[2][limit1])

        # modify
        if i == 105:
            limit3 = mf.best_ask()
            mi.modify_order_impact(order_message_id=0, new_price=limit3,
                                   new_quantity=111000)
            print('OMS: ', OMS.order_list)
            print('market, ', Market.instances['ID']._state[2][limit1])

        if i == 200:
            limit2 = mf.best_bid()
            mi.submit_order_impact(side=1, quantity=2220000, limit=limit2)
            print('second order')
            print(OMS.order_list)
            print(AgentTrade.history)
            print('market')
            print(Market.instances['ID']._state[1][limit2])

        if i == 300:
            print('afterwards')
            print(OMS.order_list)
            print(AgentTrade.history)
            print('market')
            print('bid-side')
            if limit2 in Market.instances['ID']._state[1].keys():
                print(Market.instances['ID']._state[1][limit2])
            print('ask-side')
            if limit1 in Market.instances['ID']._state[2].keys():
                print(Market.instances['ID']._state[2][limit1])

