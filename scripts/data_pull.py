import tekore as tk
from dotenv import load_dotenv
import os
import pandas as pd
import time
from tekore._model.track import Tracks
import numpy as np
from pandas.core.common import flatten


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
APP_TOKEN = tk.request_client_token(CLIENT_ID, CLIENT_SECRET)

spotify = tk.Spotify(APP_TOKEN)


def _unnest_lists(df: pd.DataFrame):
    """Used to "explode" lists in audio analysis segements dataframe."""

    for col, dtype in df.dtypes.to_dict().items():
        if dtype == object and all(hasattr(ele, "__iter__") for ele in df[col]):
            temp = df[col].apply(pd.Series)
            new_names = [f"{col}_{column}" for column in temp.columns]
            temp.rename(columns=dict(zip(temp.columns, new_names)), inplace=True)
            df = pd.concat([df, temp], axis=1)
            df.drop(columns=col, inplace=True)
    return df

def create_sample(track_id):
    results = spotify.track_audio_analysis(track_id)
    df=pd.json_normalize(results.segments.asbuiltin())
    df = _unnest_lists(df)
   
 
    return df


def get_track_ids(playlist_id):
    
    playlist = spotify.playlist(playlist_id)
    tracks = playlist.tracks.items
    return [t.track.id for t in tracks]


def get_album_ids(playlist_id):
     playlist = spotify.playlist(playlist_id)
     tracks = playlist.tracks.items
     return [t.track.album.id for t in tracks]
    
def build_segment_data(playlist_id):
    return [create_sample(track).to_numpy(dtype = np.float64) for track in get_track_ids(playlist_id)]


def get_audio_features(playlist_id):
    track_ids = get_track_ids(playlist_id)
    
    #Split into 100 track chunks
    chunks = [track_ids[i * 100:(i + 1) * 100] for i in range((len(track_ids) + 100 - 1) // 100)] 
    
    response_chunks = [spotify.tracks_audio_features(chunk) for chunk in chunks]
    return pd.DataFrame(flatten(response_chunks))

def get_album_genres(album_ids):
    
    #Split into 20 album chunks
    chunks = [album_ids[i * 20:(i + 1) * 20] for i in range((len(album_ids) + 20 - 1) // 20)]
    response_chunks = [spotify.albums(chunk) for chunk in chunks]
    albums = list(flatten(response_chunks))
   
    genres = [album.genres for album in albums]
    return dict(zip(album_ids,genres))
    

if __name__ == "__main__":
    pl = "37i9dQZF1DWYz61oV0Yc4H"

    a= get_album_ids(pl) 
    get_album_genres(a)