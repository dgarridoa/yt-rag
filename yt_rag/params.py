import argparse
import pathlib
import sys
from typing import Any

import yaml
from pydantic import BaseModel


class CommonParams(BaseModel):
    env: str
    key_vault: str = "kv-yt-rag"
    catalog: str | None = None
    database: str


class VideoCaptionsParams(CommonParams):
    channel_id: str
    timeout: int = 60
    language: str = "en"
    use_proxy: bool = False
    proxies: dict | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.use_proxy is True:
            try:
                from databricks.sdk import WorkspaceClient

                w = WorkspaceClient()
                dbutils = w.dbutils
            except ValueError:
                from databricks.sdk.runtime import dbutils

            self.proxies = {
                "https": dbutils.secrets.get(self.key_vault, "HTTP-PROXY")
            }


class ChunkParams(CommonParams):
    chunk_size: int = 500
    chunk_overlap: int = 50


class EmbeddingParams(CommonParams):
    endpoint_name: str


class VectorStoreRetrieverParams(CommonParams):
    table_url: str
    M: int = 32
    efConstruction: int = 40
    efSearch: int = 10
    k: int = 5


class ChatParams(CommonParams):
    endpoint_name: str
    temperature: float = 0.0
    max_tokens: int = 500


class Params(BaseModel):
    video_captions: VideoCaptionsParams
    chunk: ChunkParams
    embedding: EmbeddingParams
    retriever: VectorStoreRetrieverParams
    chat: ChatParams


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file")
    conf_file = parser.parse_known_args(sys.argv[1:])[0].conf_file
    config = yaml.safe_load(pathlib.Path(conf_file).read_text())
    return config
