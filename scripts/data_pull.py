import tekore as tk
from dotenv import load_dotenv
import os
import pandas as pd
import time
from tekore._client.api import artist
from tekore._model.track import Tracks
import numpy as np
from pandas.core.common import flatten


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
APP_TOKEN = tk.request_client_token(CLIENT_ID, CLIENT_SECRET)

spotify = tk.Spotify(APP_TOKEN)


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


def get_track_ids(playlist_id):

    playlist = spotify.playlist(playlist_id)
    tracks = playlist.tracks.items

    return [t.track.id for t in tracks]


def get_album_ids(playlist_id):
    playlist = spotify.playlist(playlist_id)
    tracks = playlist.tracks.items

    return [t.track.album.id for t in tracks]


def get_artist_ids(playlist_id):
    """Returns : List of tuples - (track_id , artist_ids)"""

    playlist = spotify.playlist(playlist_id)
    tracks = playlist.tracks.items
    track_ids = [t.track.id for t in tracks]
    artists = [t.track.artists for t in tracks]
    artists = [[a.id for a in artist] for artist in artists]

    return dict(zip(track_ids, artists))


def get_track_genres_by_artist(playlist_id):
    """NOTE: We need to be specific about this. An artist has genres associated with it. A track does not.\n
    For the purpose of this we will have to assume essentially that all tracks by an artist are of the \n
    same subset of genre.

    _______________________________________________________________________________________________________

    Returns: List of tuples - (track_id , genres)
    """

    track_artist_dict = get_artist_ids(playlist_id)

    artist_genres = {}
    for chunk in chunkerize(flatten(track_artist_dict.values()), 50):
        results = spotify.artists(chunk)

        artist_genres.update({artist.id: tuple(artist.genres) for artist in results})

    track_genres = {}

    # TODO
    # refactor this
    for track, artist_list in track_artist_dict.items():
        genres = set()

        for artist in artist_list:
            for genre in artist_genres[artist]:
                genres.add(genre)
        track_genres[track] = genres

    print(track_genres)

    """
    Data note : we probably will want to filter out alot of the obscure categories. We will need some way\n
    that the model views indie rock as more similar to scottish rock than it does classical.
    """


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


if __name__ == "__main__":
    pl = "37i9dQZF1DWYz61oV0Yc4H"

    # a= get_album_ids(pl)
    # d=get_album_genres(a)
    # print(d)

    get_track_genres_by_artist(pl)
