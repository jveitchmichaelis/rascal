import csv
import logging
import os
import pickle
import re
import shutil
from importlib import resources as import_resources
from typing import Dict, List, Optional
from urllib import request

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Fix for some installations where SSL requests fail
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def get_nist_rows(element, min_wave=0, max_wave=300000, unit="A"):

    sentinel_path = f"_processing_{element}_{min_wave}_{max_wave}_{unit}"
    cache_path = f"raw_{element}_{min_wave}_{max_wave}_{unit}.pkl"

    if not os.path.exists(sentinel_path) and os.path.exists(cache_path):
        logger.info(f"Using cached file for {element}")

        with open(cache_path, "rb") as fp:
            reader = pickle.load(fp)
    else:

        with open(sentinel_path, "w") as fp:
            fp.write(" ")

        logger.info(
            f"Querying NIST for {element} lines between {min_wave}{unit} and {max_wave}{unit}"
        )

        query = f"https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={element}&limits_type=0&low_w={min_wave}&upp_w={max_wave}&unit={0 if unit=='A' else 1}&de=0&I_scale_type=1&format=2&line_out=0&remove_js=on&no_spaces=on&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"

        res = request.urlopen(query)

        data_str = res.read().decode()
        reader = csv.DictReader(data_str.split("\n"), dialect="excel")

        with open(cache_path, "wb") as fp:
            pickle.dump(list(reader), fp)

        os.remove(sentinel_path)

    return reader


def toRoman(n: int) -> str:
    """Return the Roman numeral for the given integer.

    Parameters
    ----------
    n : int
        Integer to convert to Roman numeral form.

    Returns
    -------
    str
        Roman numeral(s).
    """
    # https://stackoverflow.com/a/74785461/395457
    g = {
        1: ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"],
        2: ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "LC"],
        3: ["", "C", "CC", "CCC", "CD", "DC", "DCC", "DCCC", "CM"],
        4: ["", "M", "MM", "MMM", "MMMM"],
    }
    return "".join(
        g[len(str(n)) - ind][int(s)] for ind, s in enumerate(str(n))
    )


def dump_line_list(
    line_list: List[Dict], fname: Optional[str] = "nist_clean.csv"
) -> None:
    """Dump a list of lines to a CSV file.

    Parameters
    ----------
    line_list : _type_
        List of line dictionaries. Should at least contain 'element', 'sp_num',
        'line_ref' and 'obs_wl_vac(A)' or 'obs_wl_vac(nm)' keys.
    fname : str, optional
        _description_, by default 'nist_clean.csv'
    """

    with open(fname, "w") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "element",
                "wavelength",
                "intensity",
                "acc",
                "state",
                "source",
                "reference",
                "unit",
            ],
        )
        writer.writeheader()

        for line in line_list:

            # Attempt to extract intensity
            inten_tmp = re.match("[0-9]+.[0-9]+", line["intens"].split('"')[1])

            if inten_tmp is None:
                intensity = 0
            else:
                intensity = inten_tmp.group()

            data = {
                "element": line["element"],
                "source": "nist",
                "acc": line["Acc"],
                "state": int(line["sp_num"]),
                "intensity": intensity,
                "reference": line["line_ref"].split('"')[1],
            }

            if "obs_wl_vac(A)" in line:
                data["wavelength"] = line["obs_wl_vac(A)"].split('"')[1]
                data["unit"] = "A"
            elif "obs_wl_vac(nm)" in line:
                data["wavelength"] = line["obs_wl_vac(nm)"].split('"')[1]
                data["unit"] = "nm"

            writer.writerow(data)


def create_line_list(elements: List[str]) -> List[dict]:
    """Create a line list from NIST data, all elements.

    Parameters
    ----------
    elements : List[str]
        List of elements to query.

    Returns
    -------
    List[dict]
        List of line dictionaries.

    """

    for element in tqdm(elements):
        line_list = []
        res = get_nist_rows(element)
        for line in res:
            if "element" not in line:
                continue
            line_list.append(line)

        line_list = sorted(
            line_list,
            key=lambda x: (x["element"], x["obs_wl_vac(A)"].split('"')[1]),
        )

        dump_line_list(line_list, fname=f"nist_clean_{element}.csv")

    return line_list


def get_elements():
    ref = (
        import_resources.files("rascal") / "arc_lines/pubchem_elements_all.csv"
    )

    with import_resources.as_file(ref) as path:
        with open(path) as fp:
            reader = csv.DictReader(fp.readlines(), delimiter=",")
            elements = [row["Symbol"] for row in reader]

    return elements


if __name__ == "__main__":

    # National Center for Biotechnology Information (2023). Periodic Table of Elements.
    # Retrieved March 27, 2023 from https://pubchem.ncbi.nlm.nih.gov/periodic-table/.

    create_line_list(get_elements())
