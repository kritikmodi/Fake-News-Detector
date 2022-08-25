from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(vid):
    response = YouTubeTranscriptApi.get_transcript(vid)
    l = len(response)
    print(l)
    full_text =''
    for i in range(l):
        text = response[i]['text']
        full_text += text
    return full_text

vvid = 'th5_9woFJmk'
result = get_transcript(vvid)
print(result)
