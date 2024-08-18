from datetime import datetime

import requests
from pydantic import BaseModel
from requests.models import HTTPError
from youtube_transcript_api import YouTubeTranscriptApi


class Video(BaseModel):
    channel_id: str
    video_id: str
    title: str
    publish_time: datetime

    def __hash__(self):
        return hash(self.video_id)


class Caption(BaseModel):
    text: str
    start: float
    duration: float


class VideoTranscript(BaseModel):
    video: Video
    transcript: list[Caption]


YOUTUBE_OPERATIONAL_API_URL = "https://yt.lemnoslife.com"


def get_videos_from_channel(channel_id: str, timeout: int = 5) -> set[Video]:
    URL = f"{YOUTUBE_OPERATIONAL_API_URL}/noKey/playlistItems"
    # if it has a "Videos" playlist on the main page
    playlist_id = "UULF" + channel_id[2:]
    params = {
        "part": "snippet",
        "playlistId": playlist_id,
        "maxResults": 50,
    }

    response = requests.get(URL, params=params, timeout=timeout).json()
    if "error" in response and response["error"]["status_code"] == 404:
        # if it has not a "Videos" playlist on the main page
        params["playlistId"] = "UU" + channel_id[2:]

    videos = set()
    while True:
        response = requests.get(URL, params=params, timeout=timeout).json()
        if "error" in response:
            raise HTTPError("Unable to get video.", response["error"])

        if "items" in response:
            for item in response["items"]:
                video_id = item["snippet"]["resourceId"]["videoId"]
                title = item["snippet"]["title"]
                publish_time = datetime.fromisoformat(
                    item["snippet"]["publishedAt"]
                )
                videos.add(
                    Video(
                        channel_id=channel_id,
                        video_id=video_id,
                        title=title,
                        publish_time=publish_time,
                    )
                )

        nextPageToken = response.get("nextPageToken")
        if not nextPageToken:
            break
        params["pageToken"] = nextPageToken

    return videos


def get_transcript_from_video(
    video_id: str, language: str = "en"
) -> list[Caption]:
    captions = YouTubeTranscriptApi.get_transcript(
        video_id, languages=[language]
    )
    captions = [Caption(**caption) for caption in captions]
    return captions
