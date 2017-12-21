#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
# python test.py ../sun-20171114 ../log/
##

import fnmatch
import os
import sys
import read_dump
import json

def main(dump, log, dest):
    matches = []
    for root, dirnames, filenames in os.walk(dump):
        for filename in fnmatch.filter(filenames, 'heap*'):
            matches.append(os.path.join(root, filename))
    matches.sort()
#    print matches
    data = []
    for each_file in matches:
        each_file_split = each_file.split('/')
        # binary_name = each_file_split[2]
	binary_name = each_file_split[each_file_split.index('memory') - 1]
        binary_name_split = binary_name.split('_')
        binary_name = binary_name_split[0]
        log_file = log + '/'  + binary_name + '.txt'
        print each_file
        print log_file
        ## generate the argument
        #  argument = '-f ' + each_file + ' -g ' + log_file
                 
            
        offset, bin_int = read_dump.bin_to_int(each_file)
        final_bin_int = read_dump.gen_bin_to_int_valide(bin_int)
        # print final_bin_int
        memory_obj = read_dump.get_mem_alloc(each_file, log_file, offset)
        # print memory_obj

        # use get_label_meta to replace get_label, get_label_meta will add label 2 to meta data.
        label = read_dump.get_label_meta(final_bin_int, memory_obj)
        # label = read_dump.get_label(final_bin_int, memory_obj)
        final_list = [final_bin_int, label]
        ## final_list is the result for each memory dump image and malloc log file
        #print final_list
        data.append(final_list)

    f = open(dest, 'w')
    json.dump(data, f)
    f.close()
    print "################done"
 

if __name__ == '__main__':
    main()
