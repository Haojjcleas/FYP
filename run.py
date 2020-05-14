#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Junjie Hao <CB62935@um.edu.mo>
# Created Date : May 1st 2020, 12:05:49
# Last Modified: May 7th 2020, 23:05:35
# --------------------------------------------------

import argparse
from CNC import CCNC


def main():

    classifier = CCNC()

    if arg.train:
        classifier.max_iter = MAX_ITER
        classifier.train()
        classifier.save()
    if arg.dev:
        try:
            classifier.load_model()
            classifier.beta = BETA
            classifier.test()
        except Exception as e:
            print(e)
    if arg.show:
        try:
            classifier.load_model()
            classifier.show_samples(BOUND)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='?', const=True, default=False)
    parser.add_argument('-d', '--dev', nargs='?', const=True, default=False)
    parser.add_argument('-s', '--show', nargs='?', const=True, default=False)
    arg = parser.parse_args()

    #====== Customization ======
    BETA = 0.5
    MAX_ITER = 20
    BOUND = (0, 20)
    #==========================

    main()

