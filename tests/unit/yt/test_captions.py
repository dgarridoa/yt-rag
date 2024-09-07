from datetime import datetime, timezone

from yt_rag.yt.captions import (
    Caption,
    Video,
    get_transcript_from_video,
    get_videos_from_channel,
)

channel_id = "UC34rhn8Um7R18-BHjPklYlw"
video_id = "Guy5D3PJlZk"


def test_get_videos_from_channel():
    videos = get_videos_from_channel(channel_id, 10)
    expected_videos = {
        Video(
            channel_id=channel_id,
            video_id=video_id,
            title="Agile Manifesto",
            publish_time=datetime(2024, 8, 9, 16, 3, 23, tzinfo=timezone.utc),
        )
    }
    assert videos == expected_videos


def test_transcript_from_video():
    transcript = get_transcript_from_video(video_id)
    expected_transcript = [
        Caption(
            text="I often make this joke which is agile's",
            start=0.199,
            duration=3.961,
        ),
        Caption(
            text="a lot like communism you know people",
            start=2.32,
            duration=4.8,
        ),
        Caption(
            text="just keep not trying it correctly um",
            start=4.16,
            duration=6.12,
        ),
        Caption(
            text="what is what is the correct way to",
            start=7.12,
            duration=8.08,
        ),
        Caption(
            text="Agile oh gee um it's a real simple idea",
            start=10.28,
            duration=7.52,
        ),
        Caption(
            text="right uh do things in really short",
            start=15.2,
            duration=4.72,
        ),
        Caption(
            text="sequences measure how much you get done",
            start=17.8,
            duration=4.16,
        ),
        Caption(
            text="in every sequence use that measurement",
            start=19.92,
            duration=4.0,
        ),
        Caption(
            text="to project an end date and tell", start=21.96, duration=6.04
        ),
        Caption(
            text="everybody that's kind of it", start=23.92, duration=4.08
        ),
    ]
    assert transcript == expected_transcript
