#/usr/bin/python

##
# read memory dump
# python read_dump.py -f ../sun-20171113/ls_0/memory/user/heap@01e6d000 -g ../log/ls.txt
# python read_dump.py -f ../sun-20171114/uname-a_1/memory/user/heap@024be000 -g ../log/uname-a.txt
##


import sys
import getopt
import os

'''
## parsing the argument
'''

def parsing_arg (argument):
	opts,args = getopt.getopt(argument,'f:g:')
	for o,a in opts:
		if o == '-f':
			filename = a
		if o == '-g':
			log = a
	return filename, log



'''
## read memory dump, bin --> hex --> int
## get the heap base memory address
'''
def bin_to_int(filename):
    ## get the heap base address int value to offset
	offset = 0
	basename = os.path.basename(filename)
	basename_split = basename.split('@')
	offset = int(basename_split[1], 16) ## hex to int
    ## convert memory dump bin format to int value list
	bin_int = []
	with open(filename,"rb") as f:
		block = f.read()
		for ch in block:
			ch_int = int(hex(ord(ch)), 0)
			bin_int.append(ch_int)
	return offset, bin_int



# offset, bin_int = bin_to_int(filename)
# print offset



'''
## We have got the binary set (byte by byte), we don't care about uninitialized memory aera.
'''
def gen_bin_to_int_valide(bin_int):
	count = 0
	for i in reversed(bin_int):
		if i != 0:
			return bin_int[0: (len(bin_int) - count + 1)] 
		count += 1

# final_bin_int = gen_bin_to_int_valide(bin_int)
# print final_bin_int

'''
# get memory object begging address and size
# padding == False, we don't consider padding
# meta == False, we don't conside meta data
'''
def get_mem_alloc(filename, log, offset, padding, meta):


	filename_split = filename.split('/')
	#sign_str = filename_split[2]
	sign_str = filename_split[filename_split.index("memory") - 1]

	memory_obj = {}

	f = open(log, 'r')
	line = f.readline()
	while line:
		malloc_size_num = line.find('malloc-size')
		if malloc_size_num >= 0:
			malloc_size = line[12:len(line)-1]
			malloc_size = int(malloc_size, 0)
			#print malloc_size

		malloc_addr_num = line.find('malloc-addr')
		if malloc_addr_num >= 0:
			malloc_addr = line[12:len(line)-1]
			malloc_addr = int(malloc_addr, 0)
			#print malloc_addr
			memory_obj[malloc_addr-offset] = malloc_size
	                #print memory_obj

		free_num = line.find('free')
		if free_num >= 0:
			free_addr = line[5:len(line)-1]
			free_addr = int(free_addr, 0)
			#print free_addr
			if free_addr != 0:
				del memory_obj[free_addr-offset]

		realloc_size_num = line.find('realloc-size')
	        if realloc_size_num >= 0:
			realloc_size = line[13:len(line)-1]
	                realloc_size = int(realloc_size, 0)
	                #print realloc_size

	        realloc_pre_addr_num = line.find('realloc-pre-addr')
	        if realloc_pre_addr_num >= 0:
			realloc_pre_addr = line[17:len(line)-1]
	                realloc_pre_addr = int(realloc_pre_addr, 0)
	                #print realloc_pre_addr
	                if realloc_pre_addr != 0 and (realloc_pre_addr-offset) in memory_obj:
	                	del memory_obj[realloc_pre_addr-offset]

		realloc_addr_num = line.find('realloc-addr')
	        if realloc_addr_num >= 0:
			realloc_addr = line[13:len(line)-1]
	                realloc_addr = int(realloc_addr, 0)
			#print realloc_addr
	                memory_obj[realloc_addr-offset] = realloc_size
	               # print memory_obj

		if line[0:len(line)-1] == sign_str:
			#print 'stop'
			break
		line = f.readline()
	# print memory_obj
	# return memory_obj
	if meta:
		temp_memory_obj = {}
		for key in memory_obj:
			temp_memory_obj[key - 8] = memory_obj[key] + 8
			if padding:
				remainder = temp_memory_obj[key - 8] % 16
				quot = temp_memory_obj[key - 8] / 16
				if remainder != 0:
					temp_memory_obj[key - 8] += (16 - remainder)
				if quot == 0:
					temp_memory_obj[key - 8] *= 2
		# print temp_memory_obj
		memory_obj = temp_memory_obj
	return memory_obj


'''
# If all content of memory object are zero, we don't label it as memory object
'''
def remove_zero_mem_obj(final_bin_int, memory_obj):
	rm_key = []
	for key in memory_obj:
		count = 0
		for index in range (key + 8, key+memory_obj[key]):
			if final_bin_int[index] == 0:
				count += 1
		if count == (memory_obj[key] - 8):
			# print "all content is zero", key
			rm_key.append(key)
	for item in rm_key:
		del memory_obj[item]
	# print memory_obj


'''
# Based on whether we consider meta data, we set the label
'''
def set_label(final_bin_int, memory_obj, meta):
	label = []
        #print '###', len(final_bin_int)
	for i in range (0, len(final_bin_int)):
		label.append(0)

	if meta:
		for key in memory_obj:
			#print key, memory_obj[key]
			for index in range(key, key + 8):
				label[index] = 2

			for index in range (key + 8, key+memory_obj[key]):
				label[index] = 1
	else:
		for key in memory_obj:
			#print key, memory_obj[key]
			for index in range (key, key+memory_obj[key]):
				label[index] = 1	

	return label


def main():

	filename = None
	log = None

	meta = True
	padding = False

	filename, log = parsing_arg(sys.argv[1:])
	offset, bin_int = bin_to_int(filename)
	final_bin_int = gen_bin_to_int_valide(bin_int)
	print final_bin_int
	memory_obj = get_mem_alloc(filename, log, offset, padding, meta)
	print memory_obj
	remove_zero_mem_obj(final_bin_int, memory_obj)
	label = set_label(final_bin_int, memory_obj, meta)
	final_list = [final_bin_int, label]
	print final_list




if __name__ == '__main__':
	main()	
