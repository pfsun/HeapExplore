#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
# python test.py ../sun-20171114 ../log/
##

import fnmatch
import os
import sys
import read_dump

def main():
	matches = []
	for root, dirnames, filenames in os.walk(sys.argv[1]):
    		for filename in fnmatch.filter(filenames, 'heap*'):
        		matches.append(os.path.join(root, filename))

	print len(matches)

main()
