#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2



import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def plot_lob(level_2_dict):

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