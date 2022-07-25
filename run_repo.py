import os
import sys
import argparse
from runners import run as seat_belt_detection
from runners import run as fastness_detection 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='seat belt fastness detection system')
    parser.add_argument('--seat_belts', action='store_true', help='detectes seat belts, putting bounded boxes on seat belt if exists.')
    parser.add_argument('--fastness', action='store_true', help='detectes if seat belt is fastened.')
    args = parser.parse_args()
    parser.print_help()

    if args.seat_belts:
        seat_belt_detection()

    if args.fastness:
        fastness_detection()







