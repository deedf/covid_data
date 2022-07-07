#!/usr/bin/env python3
"""
Make symptoms graph.
"""
import logging
from collections import namedtuple
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import requests
from cachecontrol import CacheControl
from cachecontrol.caches import FileCache
from requests import Response

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    cached_sess = CacheControl(
        requests.Session(), cache=FileCache(Path.home() / ".webcache")
    )
except Exception as e:  # pylint: disable=broad-except
    logger.debug(e)

PACKAGE_URL = (
    "https://ckan.opendata.swiss/api/3/action/package_show?id=covid-19-schweiz"
)

START_DATE = date(2021, 3, 23)
END_DATE = date(2022, 4, 5)
AGE_BAR_DICT = {
    "0 - 9": (0, 9),
    "10 - 19": (10, 19),
    "20 - 29": (20, 29),
    "30 - 39": (30, 39),
    "40 - 49": (40, 49),
    "50 - 59": (50, 59),
    "60 - 69": (60, 69),
    "70 - 79": (70, 79),
    "80+": (80, 89),
    "Unbekannt": (90, 99),
    "0 - 1": (0, 1),
    "12 - 17": (12, 17),
    "18 - 44": (18, 44),
    "2 - 11": (2, 11),
    "45 - 64": (45, 64),
    "65 - 74": (65, 74),
    "75+": (75, 89),
    "unknown": (90, 99),
}
CovidData = namedtuple("CovidData", ["hosp", "death", "symptoms"])


def _get_packages(url: str) -> Dict[str, Any]:
    result = cached_sess.get(url)
    result.raise_for_status()
    return result.json()["result"]


def _get_data(
    package_url: str, date_attr: str, cutoff_date: str
) -> List[Dict[str, str]]:
    result: Response = cached_sess.get(package_url)
    result.raise_for_status()
    return [
        i
        for i in result.json()
        if str(i[date_attr]) == cutoff_date and i["geoRegion"] == "CHFL"
    ]


def _get_all_data(
    death: Dict[str, Any], hosp: Dict[str, Any], symptoms: Dict[str, Any], at_date: date
) -> CovidData:
    iso_calendar = date.isocalendar(at_date)
    date_iso = str(iso_calendar[0]) + str(iso_calendar[1])
    death_data = _get_data(death["download_url"], "datum", date_iso)
    hosp_data = _get_data(hosp["download_url"], "datum", date_iso)
    symptom_data = _get_data(symptoms["download_url"], "date", at_date.isoformat())
    hosp_count = dict((i["altersklasse_covid19"], i["sumTotal"]) for i in hosp_data)
    death_count = dict((i["altersklasse_covid19"], i["sumTotal"]) for i in death_data)
    symptom_count = dict(
        [
            (i["age_group"], i["sumTotal"])
            for i in symptom_data
            if i["vaccine"] == "all"
            and i["age_group"] != "all"
            and i["severity"] == "all"
        ]
    )
    return CovidData(hosp_count, death_count, symptom_count)


def _diff(start: Dict[str, Any], end: Dict[str, Any]) -> Dict[str, Any]:
    return dict([(k, end[k] - start[k]) for k in end.keys()])


def _build_graph(
    death: Dict[str, Any], hosp: Dict[str, Any], symptoms: Dict[str, Any]
) -> None:
    start = _get_all_data(death, hosp, symptoms, START_DATE)
    end = _get_all_data(death, hosp, symptoms, END_DATE)
    counts = CovidData(
        _diff(start.hosp, end.hosp),
        _diff(start.death, end.death),
        _diff(start.symptoms, end.symptoms),
    )
    plt.rcdefaults()
    _, plot = plt.subplots()
    for age in counts.death.keys():
        death_count: int = counts.death[age]
        (y_1, y_2) = AGE_BAR_DICT[age]
        # years are 0-based so add 1 for height
        height: int = y_2 - y_1 + 1
        d_width = death_count / height
        plot.barh(
            y=y_1,
            width=-d_width,
            left=0,
            height=height,
            align="edge",
            color=(0.1, 0.2, 0.5),
            linewidth=0,
        )
        hosp_count: int = counts.hosp[age]
        (y_1, y_2) = AGE_BAR_DICT[age]
        # years are 0-based so add 1 for height
        height = y_2 - y_1 + 1
        plot.barh(
            y=y_1,
            width=-hosp_count / height,
            left=-d_width,
            height=height,
            align="edge",
            color=(0.1, 0.2, 0.8),
            linewidth=0,
        )
    for age in counts.symptoms.keys():
        count: int = counts.symptoms[age]
        (y_1, y_2) = AGE_BAR_DICT[age]
        height = y_2 - y_1
        plot.barh(
            y=y_1,
            width=count / height,
            left=0,
            height=height,
            align="edge",
            color=(0.1, 0.2, 0.7),
            linewidth=0,
        )

    plt.show()


def _main() -> None:
    packages = _get_packages(PACKAGE_URL)
    resources_json: List[Dict[str, Any]] = packages["resources"]
    resources_dict: Dict[str, Dict[str, Any]] = dict(
        [(r["identifier"], r) for r in resources_json]
    )
    _build_graph(
        resources_dict["weekly-death-age-range-json"],
        resources_dict["weekly-hosp-age-range-json"],
        resources_dict["daily-vacc-symptoms-json"],
    )


if __name__ == "__main__":
    _main()
