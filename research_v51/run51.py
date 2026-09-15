"""Manually plan or execute one R49/R50 prospective cycle using the actual clock."""
import argparse
from common51 import *
from cycle51 import Backend, cycle, describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'cycle'])
    args = parser.parse_args()
    check_freeze(True)
    backend = Backend(Journal())
    if args.command == 'plan':
        backend.validate()
        result = describe(backend)
    else:
        result = cycle(backend, RUNTIME)
    print(encoded(result).decode('utf-8'))
    if result['status'] == 'FAILED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
