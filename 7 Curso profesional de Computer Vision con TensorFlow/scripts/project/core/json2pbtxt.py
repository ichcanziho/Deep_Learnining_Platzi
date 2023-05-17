import argparse
import json

parser = argparse.ArgumentParser(description="Json to pbtxt file converter")

parser.add_argument("-j",
                    "--json_file",
                    help="json file directory",
                    type=str)
parser.add_argument("-p",
                    "--pbtxt_file",
                    help="pbtxt file directory. Defaults to the same name and directory as json_file.",
                    type=str)

args = parser.parse_args()

if args.pbtxt_file is None:
    args.pbtxt_file = args.json_file.replace(".json", ".pbtxt")


def convert_json_to_pbtxt(json_file, pbtxt_file):

    with open(json_file, "r") as data:
        json_data = json.load(data)

    with open(pbtxt_file, "w") as f:
        for label in json_data:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


if __name__ == '__main__':
    print("Input parameters are:")
    print(args.json_file, args.pbtxt_file)
    convert_json_to_pbtxt(args.json_file, args.pbtxt_file)
    print("Done")
