from argparse import ArgumentParser
from pkg_resources import get_distribution

from balaitous import Balaitous


def cli():
    """
    Simple command line interface to run the Balaitous model on one patient
    """

    parser = ArgumentParser(description='Runs the Balaitous model on one patient')
    parser.add_argument('--path', type=str, help='path to the input image')
    parser.add_argument('--age', type=int, help='age of the patient in years')
    parser.add_argument('--sex', type=int, help='sex of the patient, 1 for male, 0 for female')
    parser.add_argument('--version', default=False, action='store_true', help='display the package version')
    args = parser.parse_args()

    if args.version:
        version = get_distribution('balaitous').version
        print(f'balaitous version : {version}')
    else:
        model = Balaitous()
        p_covid, p_severe = model(args.path, age=args.age, sex=args.sex)
        print(f'Probability covid: {100*p_covid:.2f}%')
        print(f'Probability severe: {100*p_severe:.2f}%')