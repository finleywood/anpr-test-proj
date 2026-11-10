import argparse
import cv2

from anpr_proj.anpr import ANPR

def main() -> None:
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-f", "--file", help="File path to use, can be jpg or png.", required=True)
  args = argParser.parse_args()
  im = cv2.imread(args.file)
  if im is None:
    raise Exception("Failed to load image.")
  anpr = ANPR("number-plate-yolo26s.pt")
  anpr.get_plate(im)
