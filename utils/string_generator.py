#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def generate_string(*args):
    """
    Function to generate strings from input
    arguments. Used to create uniquely identifiable
    pathname.
    """
    result_string = ""
    for x in args:
        if type(x) == list:
            for y in x:
                result_string += str(y) + "_"
        else:
            result_string += str(x) + "_"

    return result_string
