# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:30:42 2018

@author: JamesChiou
"""
import csv
import sys
import numpy as np
import decimal
from multiprocessing import Pool, Manager, cpu_count
from shutil import copyfileobj
import pathlib


class ECLAT:

    min_support = 0
    inputFile = ""
    outputFile = ""

    def __init__(self):
        if(len(sys.argv) == 4):
            min_support = float(sys.argv[1])
            inputFile = str(sys.argv[2])
            outputFile = str(sys.argv[3])
            self.executeECLAT(min_support, inputFile, outputFile)
        else:
            print("InputError")

    def round_decimal(self, x, digits=0):
        # casting to string then converting to decimal
        x = decimal.Decimal(str(x))
        # rounding for integers
        if digits == 0:
            return int(x.quantize(decimal.Decimal("1"),
                                  rounding='ROUND_HALF_UP'))
        # string in scientific notation for significant digits: 1e^x
        if digits > 1:
            string = '1e' + str(-1*digits)
        else:
            string = '1e' + str(-1*digits)
        # rounding for floating points
        return float(x.quantize(decimal.Decimal(string),
                                rounding='ROUND_HALF_UP'))

    # ECLAT Algo
    def eclat(self, prefix, start, item_dict, min_count, trans, save_dict):
        while item_dict:
            while str(start) not in item_dict:
                start += 1
            # Get the first item in item_dict
            item = item_dict.pop(str(start))
            key = str(start)
            key_count = len(item)
            if key_count >= min_count:

                # Recursive do ECLAT (BFS)
                # Store other items
                suffix = {}
                for other_key, other_item in item_dict.items():
                    # Get the intersection
                    new_item = item & other_item
                    if len(new_item) >= min_count:
                        suffix[other_key] = set()
                        suffix[other_key] = suffix[other_key].union(new_item)

                self.eclat(prefix+[key], start+1, suffix, min_count, trans,
                           save_dict)

                # If has prefix key
                if prefix:
                    # Construct save_key & Delete the last comma
                    save_key = "".join(str(word)+"," for word in
                                       sorted([int(e) for e in prefix+[key]]))
                    save_key = save_key[0:-1]

                    # Store the min_support by
                    # save_dict["pattern count"]["item pattern"] = min_support
                    if(str(len(prefix+[key])) in save_dict):
                        save_dict[str(len(prefix+[key]))][save_key] = \
                            self.round_decimal(key_count/trans, 4)
                    else:
                        save_dict[str(len(prefix+[key]))] = {}
                        save_dict[str(len(prefix+[key]))][save_key] = \
                            self.round_decimal(key_count/trans, 4)

                # If no prefix key
                else:
                    # Store the min_support by
                    # save_dict["1"]["item pattern"] = min_support
                    if(str(len(prefix+[key])) in save_dict):
                        save_dict[str(len(prefix+[key]))][key] = \
                            self.round_decimal(key_count/trans, 4)
                    else:
                        save_dict[str(len(prefix+[key]))] = {}
                        save_dict[str(len(prefix+[key]))][key] = \
                            self.round_decimal(key_count/trans, 4)
        return save_dict

    def multiSort(self, save_dict, order_list, i):
        order_list1 = sorted(save_dict.items())
        order_list[i] = sorted(order_list1,
                               key=lambda s: [int(x) for x in s[0].split(",")])
        with open("temp/"+str(i), 'w', encoding='utf-8', newline='\n') as file:
            for n in order_list[i]:
                file.write(n[0]+":{:.4f}\n".format(np.around(n[1], 4)))

    # Main program
    def executeECLAT(self, min_support=0.1, inputFile="", outputFile=""):
        # Read data from inputFile
        item_dict = {}
        trans = 0
        count = 0
        with open(inputFile, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                trans += 1
                for item in row:
                    if item not in item_dict:
                        item_dict[item] = set()
                    item_dict[item].add(trans)
        # Vertical data format builded
        min_count = min_support*trans
        for key, value in item_dict.copy().items():
            if len(value) < min_count:
                item_dict.pop(key)

        save_dict = {}
        prefix = []
        start = 0
        # Do ECLAT
        self.eclat(prefix, start, item_dict, min_count, trans, save_dict)
        # Count all frequent items ( counts > min_support*transaction )
        count = 0
        for key, dict_value in save_dict.items():
            count += len(dict_value)
        # Order the save_dict to oreder_list and save to the file
        order_list = []
        m = Manager()
        order_list = m.list(order_list)
        # Multiprocessed sort & Write file
        pathlib.Path('temp').mkdir(parents=True, exist_ok=True)
        with Pool(cpu_count()-1) as p:
            for i in range(0, len(save_dict)):
                order_list.append([])
                p.apply_async(self.multiSort, args=(save_dict[str(i+1)],
                              order_list, i))
            p.close()
            p.join()
        # Concaenate files to outputFile
        with open(outputFile, 'wb') as wfd:
            for f in [str(i) for i in range(0, len(save_dict))]:
                with open("temp/"+f, 'rb') as fd:
                    copyfileobj(fd, wfd, 1024*1024*10)


if __name__ == '__main__':
    c = ECLAT()
