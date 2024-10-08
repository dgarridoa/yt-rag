import logging
import pathlib
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import requests
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_ipaddr

from yt_rag.conn import get_spark
from yt_rag.params import Params
from yt_rag.settings import get_settings
from yt_rag.vs.chat import get_chat_model
from yt_rag.vs.embedding import get_embedding_model
from yt_rag.vs.prompt import get_prompt
from yt_rag.vs.rag import create_rag_chain
from yt_rag.vs.retriever import create_vector_store_retriever

settings = get_settings()


class RAGInput(BaseModel):
    input: Annotated[
        str,
        Field(
            title="Input text",
            examples=["What is the analogy between agile and comunism?"],
        ),
    ]


class Document(BaseModel):
    id: Annotated[
        str,
        Field(
            title="Chunk id",
            examples=["UCUyeluBRhGPCW4rPe_UvBZQ_UBXXw2JSloo_13"],
        ),
    ]
    video_id: Annotated[
        str, Field(title="YouTube video id", examples=["UBXXw2JSloo"])
    ]
    title: Annotated[
        str,
        Field(
            title="YouTube video title", examples=["I Interviewed Uncle Bob"]
        ),
    ]
    publish_time: Annotated[
        datetime,
        Field(
            title="Youtube video published time",
            examples=["2024-04-29T18:10:20"],
        ),
    ]
    content: Annotated[
        str,
        Field(
            title="Chunk text",
            examples=[
                "are we sure we're on the same topic I'm pretty sure ..."
            ],
        ),
    ]


class RAGOutput(BaseModel):
    answer: Annotated[
        str,
        Field(
            title="Answer text",
            examples=["The analogy between agile and communism ..."],
        ),
    ]
    context: list[Document]


class RAGError(BaseModel):
    error: str
    response: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    spark = get_spark()
    conf_file = f"conf/{settings.env}_config.yml"
    config = yaml.safe_load(pathlib.Path(conf_file).read_text())

    params = Params(**config)

    chat_model = get_chat_model(
        endpoint_name=params.chat.endpoint_name,
        temperature=params.chat.temperature,
        max_tokens=params.chat.max_tokens,
    )
    embedding_function = get_embedding_model(
        endpoint_name=params.embedding.endpoint_name
    )
    df = (
        spark.read.format("deltaSharing")
        .load(params.retriever.table_url)
        .toPandas()
    )
    retriever = create_vector_store_retriever(
        params.retriever, df, embedding_function
    )
    prompt = get_prompt()
    app.state.rag_chain = create_rag_chain(chat_model, retriever, prompt)
    yield
    spark.stop()


limiter = Limiter(key_func=get_ipaddr)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
security = HTTPBasic()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    username = settings.api_username.get_secret_value()
    password = settings.api_password.get_secret_value()

    if username is None or password is None:
        raise ValueError(
            "API_USERNAME and API_PASSWORD environment variables must be set"
        )

    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = username.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = password.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/readiness")
async def readiness():
    return {"status": "ok"}


@app.post("/rag")
@limiter.limit("5/minute")
async def rag(
    input: str,
    username: Annotated[str, Depends(get_current_username)],
    request: Request,
) -> RAGOutput:
    response = app.state.rag_chain.invoke({"input": input})
    output = RAGOutput(
        answer=response["answer"],
        context=[
            Document(
                id=doc.metadata["id"],
                video_id=doc.metadata["video_id"],
                title=doc.metadata["title"],
                publish_time=doc.metadata["publish_time"],
                content=doc.page_content,
            )
            for doc in response["context"]
        ],
    )
    return output


def get_rag_response(
    url: str, api_username: str, api_password: str, content: str
) -> RAGOutput | RAGError:
    try:
        response = requests.post(
            url,
            auth=(
                api_username,
                api_password,
            ),
            params={"input": content},
            timeout=120,
        )
        try:
            response.raise_for_status()
            return RAGOutput.model_validate(response.json())
        except requests.exceptions.HTTPError as http_err:
            return RAGError(error=str(http_err), response=response.json())
    except Exception as err:
        return RAGError(error=str(err))


class ReadinessFilter(logging.Filter):
    def filter(self, record):
        endpoints = ["/readiness"]
        return not any(
            endpoint in record.getMessage() for endpoint in endpoints
        )


if __name__ == "__main__":
    import uvicorn

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    uvicorn.run(
        "yt_rag.api.app:app",
        host="0.0.0.0",  # noqa
        port=5000,
        log_config="log_config.yml",
    )
