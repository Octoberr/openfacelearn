# coding:utf-8

import argparse

# parser = argparse.ArgumentParser(description='somethong to learn')
# parser.add_argument('--square', help="qiu pingfang", type=int)
# parser.add_argument('--cubic', help="qiu lifang", type=int)
# args = parser.parse_args()
# if args.square:
#     print args.square**2
# if args.cubic:
#     print args.cubic**3
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                   const=sum, default=max,
                   help='sum the integers (default: find the max)')

args = parser.parse_args()
print args.accumulate(args.integers)