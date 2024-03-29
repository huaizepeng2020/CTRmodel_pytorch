from base_model.features import Number, Category, Sequence, Features
from base_model.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)
from base_model.pytorch import WideDeep


from .utils import prepare_dataloader


def test_normal():
    number_features = [
        Number('userAge', StandardScaler()),
        Number('rating', StandardScaler())]

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId', CategoryEncoder(min_cnt=1)),
        Category('topGenre', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    wide_features = ['rating', 'title', 'genres']
    deep_features = ['userAge', 'rating', 'userId', 'movieId', 'topGenre',
                     'clickedMovieIds', 'clickedMovieTopGenres']
    cross_features = [('movieId', 'clickedMovieIds'),
                      ('topGenre', 'clickedMovieTopGenres')]

    dataloader, _ = prepare_dataloader(features)

    model = WideDeep(
        features, wide_features, deep_features, cross_features,
        num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_without_number_feature():
    number_features = []

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId', CategoryEncoder(min_cnt=1)),
        Category('topGenre', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    wide_features = ['title', 'genres']
    deep_features = ['userId', 'movieId', 'topGenre',
                     'clickedMovieIds', 'clickedMovieTopGenres']
    cross_features = [('movieId', 'clickedMovieIds'),
                      ('topGenre', 'clickedMovieTopGenres')]

    dataloader, _ = prepare_dataloader(features)

    model = WideDeep(
        features, wide_features, deep_features, cross_features,
        num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_without_category_feature():
    number_features = []

    category_features = []

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    wide_features = ['title', 'genres']
    deep_features = ['clickedMovieIds', 'clickedMovieTopGenres']

    dataloader, _ = prepare_dataloader(features)

    model = WideDeep(
        features, wide_features, deep_features, [],
        num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_only_with_number_features():
    number_features = [
        Number('userAge', StandardScaler()),
        Number('rating', StandardScaler())]

    category_features = []

    sequence_features = []

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    wide_features = ['rating', 'userAge']

    dataloader, _ = prepare_dataloader(features)

    model = WideDeep(
        features, wide_features, [], [],
        num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))
