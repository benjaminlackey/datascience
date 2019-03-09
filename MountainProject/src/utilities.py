"""Utilities for making recommendations from a trained model.
"""

import numpy as np
import pandas as pd

def make_clickable(url):
    """Open in new tab using: target="_blank"
    """
    return '<a href="{}" target="_blank">{}</a>'.format(url, url)


def recommend_top_n_routes(uid, algo, df_items, n=10):
    """Recommend top N routes for a specific user.

    Parameters
    ----------
    uid : Raw id for user u.
    algo : KNN recommender.
    df_items : pd.DataFrame containing route info.
    n : Number of routes to recommend

    Returns
    -------
    df : pd.DataFrame of top n recommended routes for user uid.
    """
    # List of route ids
    iids = df_items['iid'].unique()

    # Rating of each item
    # algo.predict() takes the raw uid and raw iid
    ratings = [{'iid':iid, 'rating_est':algo.predict(uid, iid).est} for iid in iids]
    df = pd.DataFrame(ratings)

    # Get top n recommendations
    df = df.sort_values('rating_est', ascending=False).head(n)

    # Merge with all information about route
    df = pd.merge(df, df_items, on='iid')

    # Route info
    df = df.drop('description', axis=1)
    return df


def rank_partners_by_similarity(raw_uid, algo, df_partners):
    """Rank potential partners by similarity.

    Parameters
    ----------
    raw_uid : Raw id for user u.
    algo : KNN recommender.
    df_partners : pd.DataFrame containing potential partners.

    Returns
    -------
    df_ranked_partners : pd.DataFrame of partners ranked by similarity.
    """
    # Raw id of the potential partners
    raw_uid_partners = df_partners['uid'].values

    # Inner ids of the person and all potential partners
    inner_uid = algo.trainset.to_inner_uid(raw_uid)
    inner_uid_partners = np.array([algo.trainset.to_inner_uid(i) for i in raw_uid_partners])

    # Extract similarity from the similarity matrix.
    # Similarity matrix is symmetric, and is indexed by inner_uid.
    similarity = algo.sim[inner_uid, inner_uid_partners]
    df_similarity = pd.DataFrame(data={
        'uid':raw_uid_partners,
        'inner_uid':inner_uid_partners,
        'similarity':similarity})

    # Join partners data with their similarity
    df_ranked_partners = pd.merge(df_partners, df_similarity, on='uid')

    # Rank partners by similarity
    df_ranked_partners.sort_values(by='similarity', ascending=False, inplace=True)

    return df_ranked_partners


def rank_routes_by_similarity(raw_iid, algo, df_routes):
    """Rank routes by similarity.

    Parameters
    ----------
    raw_iid : Raw id for item i.
    algo : KNN recommender.
    df_routes : pd.DataFrame containing routes.

    Returns
    -------
    df_ranked_routes : pd.DataFrame of routes ranked by similarity.
    """
    # Raw id of the potential partners
    raw_iid_routes = df_routes['iid'].values

    # Inner ids of the person and all potential partners
    inner_iid = algo.trainset.to_inner_iid(raw_iid)
    inner_iid_routes = np.array([algo.trainset.to_inner_iid(i) for i in raw_iid_routes])

    # Extract similarity from the similarity matrix.
    # Similarity matrix is symmetric, and is indexed by inner_uid.
    similarity = algo.sim[inner_iid, inner_iid_routes]
    df_similarity = pd.DataFrame(data={
        'iid':raw_iid_routes,
        'inner_iid':inner_iid_routes,
        'similarity':similarity})

    # Join partners data with their similarity
    df_ranked_routes = pd.merge(df_routes, df_similarity, on='iid')

    # Rank partners by similarity
    df_ranked_routes.sort_values(by='similarity', ascending=False, inplace=True)

    return df_ranked_routes
