import argparse


def main():
    
    arg_parser = argparse.ArgumentParser(description="Neural Machine Translation Testing")
    arg_parser.add_argument("--model_file", required=True, help="Model File")

    args = arg_parser.parse_args()
    args = vars(args)
    print(args)


if __name__ == "__main__":
    main()