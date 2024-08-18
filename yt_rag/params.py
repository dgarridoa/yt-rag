import argparse
import pathlib
import sys

import yaml
from pydantic import BaseModel


class CommonParams(BaseModel):
    env: str
    catalog: str | None = None
    database: str


class VideoCaptionsParams(CommonParams):
    channel_id: str


class ChunkParams(CommonParams):
    chunk_size: int = 500
    chunk_overlap: int = 50


class Params(BaseModel):
    video_captions: VideoCaptionsParams
    chunk: ChunkParams


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file")
    conf_file = parser.parse_known_args(sys.argv[1:])[0].conf_file
    config = yaml.safe_load(pathlib.Path(conf_file).read_text())
    return config
