#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:28:39 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
split a table by line, with header
=================================== input =====================================
head1,head2,head3
line1
line2
line3
line4
=================================== output ====================================
when lines_per_file=2
file1:
head1,head2,head3
line1


file2:
head1,head2,head3
line2
line3

file3:
head1,head2,head3
line4

================================= parameters ==================================

=================================== example ===================================
python3 this_script.py retrorules_rr01_rp2_flat_all.csv 10000
=================================== warning ===================================

the 2nd to last output files lines = lines_per_file+1,
while the 1st output file lines=lines_per_file (see output)

####=======================================================================####
"""
import sys

#dp=r"retrorules_rr01_rp2_flat_all.csv"
#lines_per_file = 300

dp=sys.argv[1]
lines_per_file=int(sys.argv[2])

with open(dp, "r") as file:
    header = file.readline()

smallfile = None
file_num=0
with open(dp) as bigfile:
    for line_num, line in enumerate(bigfile):
        if line_num % lines_per_file == 0:
            if smallfile:
                smallfile.close()
                file_num+=1
            small_filename = str(dp+"_"+str(file_num))
            smallfile = open(small_filename, "w")
            if line_num!=0:
                smallfile.write(header)
        smallfile.write(line)
    if smallfile:
        smallfile.close()










