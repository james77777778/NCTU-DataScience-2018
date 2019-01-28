# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:03:07 2018

@author: JamesChiou
"""
import sys
comparison_file = sys.argv[1]
test_file = sys.argv[2]

check = True
with open(comparison_file, newline='') as file:
    with open(test_file, newline='') as file2:
        for line, line2 in zip(file, file2):
            if not line.replace('\r\n', '\n') == line2.replace('\r\n', '\n'):
                # print("fail")
                check = False
print(check)
