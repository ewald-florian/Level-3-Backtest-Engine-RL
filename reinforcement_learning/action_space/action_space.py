# Translate the output of the policy into actions
# E.g. 1 => submit market sell order (or whatever)...
"""
ActionSpace should translate the action obtained from the policy
into executable actions.

Each agent should have its own ActionSpace class which he can for example
inherit (there is no universal action space, action spaces can look very
different), the easiest way would be if RLAgennt and ActionSpace are in the
same py file and RLAgent inherits action.

E.g. if action=1 -> submit market-buy order with quantity 100

It would make sense if ActionSpace can directly access the MarketInterface
such that the actions are actually:

-> coming from the policy

-> parsed by the ActionSpace calling the MarketInterface

-> Executed by the market interface by sending orders.

How does the action get into the ActionSpace?

Option 1: via replay_episode

    environment.step(action)

        replay_episode.step(action)

            actionspace.take_action(action)

                    market_interface(parsed action)


Option 2: Action is stored in a Action class attribute: -> Action class

    environment.step(action)

        Action(action)
        replay_episode.step()

            action_space.take_action(Action.action)

                market_interface(parsed_action)


"""

