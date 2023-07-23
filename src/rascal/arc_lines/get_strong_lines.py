import argparse
import csv

from preprocess_nist import get_elements, toInt, toRoman

from rascal.util import air_to_vacuum_wavelength


def get_strong_url(name):
    return f"https://physics.nist.gov/PhysRefData/Handbook/Tables/{name.lower()}table2_a.htm"


import os
from urllib import request

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

elements = get_elements(short_name=False)
elements_short = get_elements()


def process_raw(element, data):
    """
    Process a dump of NIST's "strong lines" web page. Unfortunately no machine readable
    format is provided and the rows have variable column lengths
    so we have to bodge around it.

    """
    mode = "air"

    out = []

    for line in data:

        line = line.strip()
        cols = line.split(",")

        if len(cols) == 1:
            if "vacuum" in cols[0].lower():
                mode = "vacuum"
            elif "air" in cols[0].lower():
                mode = "air"
        else:
            intensity = 0
            note = []
            if cols[0] == "Intensity":
                continue
            # If there are only 4 columns, then first item is wavelength
            # otherwise it should always be intensity
            elif len(cols) == 3:
                wavelength, element, ref = cols

                ionisation = "I"
            elif len(cols) == 4:
                wavelength, element, ionisation, ref = cols
            else:

                try:
                    intensity = cols[0]
                    intensity = int(str(intensity).replace("*", ""))
                except:
                    continue

                # Grab single letter "other info"
                cols = cols[1:]
                while len(cols[0]) == 1:
                    note.append(cols[0])
                    cols = cols[1:]

                try:
                    wavelength, element, ionisation, ref = cols[:4]
                except:
                    continue

            wavelength = float(wavelength)

            if mode == "air":
                wavelength = air_to_vacuum_wavelength(wavelength)

            # Some of the state info is odd
            state_int = -1
            try:
                state_int = toInt(ionisation)
            except:
                pass

            out.append(
                {
                    "element": element,
                    "intensity": intensity,
                    "note": "".join(note),
                    "wavelength": wavelength,
                    "state": state_int,
                    "reference": ref,
                    "note": "".join(note),
                    "acc": "",
                    "source": "",
                    "unit": "A",
                }
            )

    return out


def main(args):

    os.makedirs("strong_lines", exist_ok=True)

    if args.download:
        for element in tqdm(elements):

            try:
                res = request.urlopen(get_strong_url(element))
            except request.HTTPError:
                print(f"Failed to download {element}")

            data_str = res.read().decode()
            soup = BeautifulSoup(data_str)

            try:
                with open(f"strong_lines/{element}.txt", "w") as fp:
                    for row in soup.select("pre")[0].text.split("\r\n"):
                        fp.write(",".join(row.split()))
                        fp.write("\n")
            except (TypeError, IndexError):
                print(f"Failed to download {element}")

    elements_short = get_elements()
    for idx, element in enumerate(elements):
        file = f"strong_lines/{element}.txt"

        if not os.path.exists(file):
            continue

        with open(file, "r") as fp:
            data = fp.readlines()

            out = process_raw(elements_short[idx], data)

        states = set([line["state"] for line in out if line["state"] >= 1])

        for state in states:
            with open(
                f"strong_lines/{elements_short[idx]}_{toRoman(state)}.csv", "w"
            ) as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "element",
                        "wavelength",
                        "intensity",
                        "acc",
                        "state",
                        "source",
                        "note",
                        "reference",
                        "unit",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [line for line in out if line["state"] == state]
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    main(args)
