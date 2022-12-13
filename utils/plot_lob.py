#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

__author__ = 'florian'

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_lob(level_2_dict):
    """Plot level 2 representation."""

    plt.style.use("dark_background")

    bid = level_2_dict[1]
    ask = level_2_dict[2]

    bid_quantities = list(bid.values())
    ask_quantities = list(ask.values())
    bid_prices = list(bid.keys())
    ask_prices = list(ask.keys())
    midpoint = int((max(bid_prices) + min(ask_prices))/2)

    prices = np.around(np.array(bid_prices + ask_prices)*1e-8, 2) # bid < ask
    quantities = np.array(bid_quantities + ask_quantities)*1e-4


    colors = ['g']*len(bid_prices) + ['r']*len(bid_prices)

    objects = prices
    y_pos = np.arange(len(objects))
    performance = quantities

    plt.barh(y_pos, performance, align='center', alpha=0.5, color = colors)
    plt.yticks(y_pos, objects)
    plt.ylabel('price level')
    plt.xlabel('aggregated quantity')
    plt.grid(linestyle='-', linewidth=0.2, color='0.8')
    plt.title('limit order book')

    return plt.show()


def plot_lob_l3(level_3_dict):
    """Plot level 3 representation."""

    l3 = level_3_dict
    all_lists = []
    labels = []
    for side in [1, 2]:
        # Reverse = True -> descending, for side == 1
        # Reverse = False -> ascending, for side == 2
        reverse_indicator = True if side == 1 else False
        for key in sorted(l3[side].keys(), reverse=reverse_indicator):
            labels.append(key)
            quantities = [d['quantity']/1_0000 for d in l3[side][key]]
            locals()['qts' + str(side) + '_' + str(key)] = quantities
            # Append list to all lists.
            all_lists.append(locals()['qts' + str(side) + '_' + str(key)])

    # Max list len.
    lengths = [len(l) for l in all_lists]
    maximum_length = max(lengths)
    # Extend lists with zeros to have matching lengths.
    for l in all_lists:
        diff = maximum_length - len(l)
        l.extend([0] * diff)

    # Prices as Labels.
    labels = np.around(np.array(labels)*1e-8, 2)
    labels = [str(label) for label in labels]

    # Create list for all order priorities.
    all_order_lists = []
    all_order_names = []
    for i in range(maximum_length):
        #locals()['order_' + str(i+1)] = [l[i] for l in all_lists]
        locals()['priority_' + str(i+1)] = [l[i] for l in all_lists]
        all_order_lists.append(locals()['priority_' + str(i+1)])
        all_order_names.append('priority ' + str(i+1))

    # -- Create Plot.

    # Prepare random colors:
    colors = ['b', 'y', 'c', 'm', 'w', 'g', 'r', 'darkviolet', 'lime', 'cornflowerblue']
    if len(all_order_lists) > 7:
        random.seed(75)
        for j in range(len(all_order_lists)-7):
            color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
            colors.append(color)

    fig, ax = plt.subplots()
    next_left = np.zeros(20)

    for i, order_list in enumerate(all_order_lists):
        ax.barh(labels, order_list, label=all_order_names[i], height=0.6, left=next_left,
                    align='center', alpha=0.7, color = colors[i])
        next_left = next_left + np.array(order_list)

    ax.set_xlabel('stacked quantities of individual orders')
    ax.set_ylabel('price level')
    ax.set_title('Level-3 Limit Order Book')
    plt.grid(linestyle='-', linewidth=0.2, color='0.8')

    labels_per_side = int(len(labels)/2)
    price_colors = ['g']*labels_per_side + ['r']*labels_per_side
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), price_colors):
        ticklabel.set_color(tickcolor)
    ax.legend(fontsize=8)

    return plt.show()


def plot_level_2_and_level_3(level_2_dict, level_3_dict, timestamp=None,
                             save_file=False):
    """Plot level 2 and level 3 representation in a subplot."""
    plt.style.use('dark_background')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

    # -- Level 2 Plot

    bid = level_2_dict[1]
    ask = level_2_dict[2]

    bid_quantities = list(bid.values())
    ask_quantities = list(ask.values())
    bid_prices = list(bid.keys())
    ask_prices = list(ask.keys())
    midpoint = int((max(bid_prices) + min(ask_prices)) / 2)
    prices = bid_prices + ask_prices
    prices = ['{:.2f}'.format(round(p * 1e-8, 2)) for p in prices]
    # prices = np.around(np.array(bid_prices + ask_prices)*1e-8, 2) # bid < ask
    quantities = np.array(bid_quantities + ask_quantities) * 1e-4

    colors = ['g'] * len(bid_prices) + ['r'] * len(bid_prices)

    objects = prices
    y_pos = np.arange(len(objects))
    performance = quantities

    ax1.barh(y_pos, performance, align='center', alpha=0.5, color=colors,
             height=0.6)
    ax1.set_yticks(y_pos, objects)
    ax1.set_ylabel('price level')
    ax1.set_xlabel('aggregated quantities')
    ax1.grid(linestyle='-', linewidth=0.2, color='0.8')
    ax1.set_title('Level-2 LOB Representation', fontsize=11)
    labels_per_side = int(len(bid_quantities))
    price_colors = ['g'] * labels_per_side + ['r'] * labels_per_side
    # TODO: colors
    for ticklabel, tickcolor in zip(ax1.get_yticklabels(), price_colors):
        ticklabel.set_color(tickcolor)

    # Add custom legend.
    red_patch = mpatches.Patch(color='brown', label='Ask Levels')
    blue_patch = mpatches.Patch(color='darkgreen', label='Bid Levels')

    ax1.legend(handles=[red_patch, blue_patch], fontsize=7, loc='lower right',
               frameon=True)

    # -- Level 3 Plot

    l3 = level_3_dict

    # -- Data

    all_lists = []
    labels = []
    for side in [1, 2]:
        # Reverse = True -> descending, for side == 1
        # Reverse = False -> ascending, for side == 2
        reverse_indicator = True if side == 1 else False
        for key in sorted(l3[side].keys(), reverse=reverse_indicator):
            labels.append(key)
            quantities = [d['quantity'] / 1_0000 for d in l3[side][key]]
            locals()['qts' + str(side) + '_' + str(key)] = quantities
            # Append list to all lists.
            all_lists.append(locals()['qts' + str(side) + '_' + str(key)])

    # Max list len.
    lengths = [len(l) for l in all_lists]
    maximum_length = max(lengths)
    # Extend lists with zeros to have matching lengths.
    for l in all_lists:
        diff = maximum_length - len(l)
        l.extend([0] * diff)

    # Prices as Labels.
    labels = ['{:.2f}'.format(round(l * 1e-8, 2)) for l in labels]

    # Create list for all order priorities.
    all_order_lists = []
    all_order_names = []
    for i in range(maximum_length):
        # locals()['order_' + str(i+1)] = [l[i] for l in all_lists]
        locals()['priority_' + str(i + 1)] = [l[i] for l in all_lists]
        all_order_lists.append(locals()['priority_' + str(i + 1)])
        all_order_names.append('Priority ' + str(i + 1))

    # -- Create Plot.

    # Prepare random colors:
    colors = ['b', 'y', 'c', 'm', 'r', 'g', 'w', 'darkviolet', 'lime',
              'cornflowerblue']
    if len(all_order_lists) > 7:
        random.seed(75)
        for j in range(len(all_order_lists) - 7):
            color = ["#" + ''.join(
                [random.choice('ABCDEF0123456789') for i in range(6)])]
            colors.append(color)

    next_left = np.zeros(20)

    for i, order_list in enumerate(all_order_lists):
        ax2.barh(labels, order_list, label=all_order_names[i], height=0.6,
                 left=next_left,
                 align='center', alpha=0.7, color=colors[i])
        next_left = next_left + np.array(order_list)

    ax2.set_xlabel('stacked quantities of individual orders')
    # ax2.set_ylabel('price level')
    ax2.set_title('Level-3 LOB Representation', fontsize=11)

    labels_per_side = int(len(labels) / 2)
    price_colors = ['g'] * labels_per_side + ['r'] * labels_per_side
    for ticklabel, tickcolor in zip(ax2.get_yticklabels(), price_colors):
        ticklabel.set_color(tickcolor)

    ax2.legend(fontsize=7, loc='lower right', frameon=True)

    fig.suptitle('Comparison of Data-Levels', fontsize=16)
    plt.grid(linestyle='-', linewidth=0.2, color='0.8')

    anotation_text = "BAYER AG LOB Snapshot | ISIN: DE000BAY0017 | Timestamp: {}".format(
        timestamp)
    plt.figtext(0.68, -0.02, anotation_text, ha="right", fontsize=10,
                color='grey')  # , #bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Note: saving to pdf yields highest possible resolution (higher than png)
    if save_file:
        plt.savefig(
            '/Users/florianewald/Desktop/comparison_of_data_levels_x.pdf')
    return plt.show()