# Homework 2 - Find Frequent Patterns

## Packages:
- Python 3.6

## Usage:
- 找出機率>min_support的Frequent Patterns
    ```bash
    $ python main.py [min_support] [inputFile(測資)] [outputFile]
    ```
    - Format:
        - 一行為一組frequent pattern
        - item數量少的pattern在前，數量多的在後
        - item數量相等的patterns，item編號起始小者在前…以此類推
        - Pattern內item編號由小到大
        - Example:
        ```bash
        0:0.2086
        1:0.1022
        2:0.2098
        3:0.1036
        4:0.2244
        5:0.4414
        6:0.3342
        7:0.3284
        8:0.2160
        9:0.5538
        10:0.4386
        ...
        ```
