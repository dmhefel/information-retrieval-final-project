from typing import Dict, Union, List, Generator, Iterator
import functools
import json
import re
import os
import time
from datetime import datetime as dt
from pathlib import Path


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_t = time.perf_counter()
        f_value = func(*args, **kwargs)
        elapsed_t = time.perf_counter() - start_t
        mins = elapsed_t // 60
        print(
            f"'{func.__name__}' elapsed time: {mins} minutes, {elapsed_t - mins * 60:0.2f} seconds"
        )
        return f_value

    return wrapper_timer


def load_wapo(wapo_jl_path: Union[str, os.PathLike]) -> Iterator[Dict]:
    """
    It should be similar to the load_wapo in HW3 with two changes:
    - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
    - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datatime package to do this.
    """
    id = "doc_id"
    title = "title"
    author = "author"
    content_str = "content_str"
    with open(wapo_jl_path) as file:
        line = file.readline()
        while line:
            json_line = json.loads(line)
            data_dict = {id: json_line[id], author: json_line[author],
                         title: json_line[title], content_str: json_line[content_str]}
            yield data_dict

def load_clean_wapo_with_embedding(
    wapo_jl_path: Union[str, os.PathLike]
) -> Generator[Dict, None, None]:
    """
    load wapo docs as a generator
    :param wapo_jl_path:
    :return: yields each document as a dict
    """
    with open(wapo_jl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield json.loads(line)


if __name__ == "__main__":
    pass
    # # Code to test the methods inside this class
    # data_dir = Path("data")
    # wapo_path = data_dir.joinpath("subset_wapo_50k_sbert_ft_filtered.jl")
    # item = load_wapo(wapo_path)
    # for i in range(5):
    #     print(next(item).keys())
    #     print(next(item).values())

