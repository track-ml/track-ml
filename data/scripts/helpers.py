from logging import exception
import tekore as tk
from dotenv import load_dotenv
import os
import pandas as pd
import time
from tekore._client.api import artist, playlist
from tekore._model.track import Tracks
import numpy as np
from pandas.core.common import flatten
from itertools import chain
import requests

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

AUTH_URL = "https://accounts.spotify.com/api/token"
BASE_URL = "https://api.spotify.com/v1"


spotify = tk.Spotify()

def get_token():

    # POST
    auth_response = requests.post(
        AUTH_URL,
        {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )

    # convert the response to JSON
    auth_response_data = auth_response.json()

    # save the access token
    return auth_response_data["access_token"]

TOKEN = get_token()
HEADERS = {"Authorization":f"Bearer {TOKEN}"}

spotify = tk.Spotify(token = TOKEN)

def get_request(endpoint, uri, token):
    headers = {"Authorization":f"Bearer {token}"}
    r = requests.get(
    "/".join(
        [
            BASE_URL,
            endpoint,
            uri
        ]
    ),
    headers=headers)
    if r.status_code == 200:
        return r
    raise r.raise_for_status()



def get_track_ids(playlist_id):

    trackset = set()
    r = get_request("playlists",playlist_id,TOKEN).json()
    
   
    r = r["tracks"]
    
   
    while 1:


        
        if len(r["items"]) < 100:
            break
        try:
            tracks = (t["track"]["id"] for t in r["items"])
            trackset.add(tracks)
        #If response is None, this will throw type error
        except TypeError:
            print("Broke early due to None response")
            break
        r = requests.get(r["next"], headers=HEADERS)
        if r.status_code == 200:
            r = r.json()
            continue
        raise r.raise_for_status()


    
    return list(flatten(trackset))


def chunkerize(iterable, chunksize: int):
    """Split an iterable into smaller chunks of a given size."""
    iterable = list(iterable)

    # if any(not hasattr(iterable,attr) for attr in ["__iter__","__len__"]):
    #     raise TypeError("msg")

    return [
        iterable[i * chunksize : (i + 1) * chunksize]
        for i in range((len(iterable) + chunksize - 1) // chunksize)
    ]


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
    df = pd.json_normalize(results.segments.asbuiltin())
    df = _unnest_lists(df)

    return df


# TODO
# #Redo this without using tekore



def get_album_ids(playlist_id):
    
    tracks = playlist.tracks.items

    return [t.track.album.id for t in tracks]


def get_artist_ids(playlist_id):
    """Returns : List of tuples - (track_id , artist_ids)"""

    r = get_request("playlists", playlist_id, TOKEN)
    if r.status_code != 200:
        raise r.raise_for_status()

    
    tracks = r.json()["tracks"]["items"]
    track_ids = [t["track"]["id"] for t in tracks]
    artists = [t["track"]["artists"] for t in tracks]
    artists = [[a["id"] for a in artist] for artist in artists]

    return dict(zip(track_ids, artists))


def get_track_genres_by_artist(playlist_id):
    """NOTE: We need to be specific about this. An artist has genres associated with it. A track does not.\n
    For the purpose of this we will have to assume essentially that all tracks by an artist are of the \n
    same subset of genre.
    \n_______________________________________________________________________________________________________
    Returns: Dict - (track_id , genres)
    """

    track_artist_dict = get_artist_ids(playlist_id)
    track_genres = {}
    artist_genres = {}
    for chunk in chunkerize(flatten(track_artist_dict.values()), 50):
        results = spotify.artists(chunk)

        artist_genres.update({artist.id: tuple(artist.genres) for artist in results})

    # TODO
    # refactor this
    for track, artist_list in track_artist_dict.items():
        genres = set()

        for artist in artist_list:
            for genre in artist_genres[artist]:
                genres.add(genre)
        track_genres[track] = genres

    return track_genres
    # Data note : we probably will want to filter out alot of the obscure categories. We will need some way\n
    # that the model views indie rock as more similar to scottish rock than it does classical.


def build_segment_data(playlist_id):

    # TODO Label segment data against genres associated with albums
    return [
        create_sample(track).to_numpy(dtype=np.float64)
        for track in get_track_ids(playlist_id)
    ]


def get_audio_features(playlist_id):
    track_ids = get_track_ids(playlist_id)

    # Split into 100 track chunks
    chunks = chunkerize(track_ids, 100)

    response_chunks = [spotify.tracks_audio_features(chunk) for chunk in chunks]
    
    
    return pd.DataFrame(flatten(response_chunks))


def get_album_genres(album_ids):

    # Split into 20 album chunks
    chunks = chunkerize(album_ids, 20)
    response_chunks = [spotify.albums(chunk) for chunk in chunks]
    albums = list(flatten(response_chunks))

    genres = [list(album.genres) for album in albums]
    return list(zip(album_ids, genres))


# TODO recode this
# This function is written like absolute shite
def build_audio_features_data(playlist_list: list):
    """Build a one-hot encoded dataframe for audio features."""

    playlist_dict_list = [
        get_track_genres_by_artist(playlist_id) for playlist_id in playlist_list
    ]
    track_genres = {k: v for x in playlist_dict_list for k, v in x.items()}
    genres = set(chain.from_iterable(track_genres.values()))
    genre_frame = pd.DataFrame(columns=list(genres))

    for id, genre_list in track_genres.items():
        for genre in genre_list:
            genre_frame.loc[id, genre] = 1
    genre_frame.drop_duplicates()

    audio_features = pd.concat(
        [get_audio_features(playlist_id) for playlist_id in playlist_list]
    )
    
    
    print(audio_features)
    audio_features.drop(
        columns=["analysis_url", "track_href", "type", "uri"], inplace=True
    )
    audio_features.set_index("id", inplace=True)
    audio_features.drop_duplicates(inplace=True)

    return genre_frame.merge(
        audio_features, how="inner", left_index=True, right_index=True
    )


def aggregate_genres(track_audio_data):    
    
  
    
    genre_cols =[
            col
            for col in track_audio_data.columns
            if track_audio_data[col].isnull().any()
        ]

    audio_cols = [col for col in track_audio_data.columns if col not in genre_cols ]
    
    data = pd.DataFrame(
        columns=audio_cols
    )
    for col in genre_cols:
    
        data.loc[col] = track_audio_data[track_audio_data[col].notnull()].aggregate(
            func="mean"
            )
    return data


if __name__ == "__main__":
    pl ="4AoI1VZwQsSKFXSQnAgxtV"
    playlist_list = [pl]
    audio = build_audio_features_data(playlist_list)
    tracks = get_track_ids(pl)
    print(aggregate_genres(audio))
    # df = get_audio_features(pl)
    # print(df)