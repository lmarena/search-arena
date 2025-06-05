import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from functools import partial
from scipy.special import expit
from scipy.optimize import minimize

########################################
# Battle Analysis Utils
########################################

def compute_pairwise_win_fraction(
    battles: pd.DataFrame,
    model_order: Optional[List[str]],
    limit_show_number: Optional[int] = None
) -> pd.DataFrame:
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (
        num_battles_ptbl + num_battles_ptbl.T
    )
    if model_order is None:
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        model_order = list(prop_wins.keys())
    if limit_show_number is not None:
        model_order = model_order[:limit_show_number]
    row_beats_col = row_beats_col_freq.loc[model_order, model_order]
    return row_beats_col


def get_median_elo_from_bootstrap(
    bootstrap_df: pd.DataFrame
) -> Dict[Any, int]:
    median = dict(bootstrap_df.quantile(0.5))
    median = {k: int(v + 0.5) for k, v in median.items()}
    return median


def get_matchups_models(
    df: pd.DataFrame
) -> Tuple[np.ndarray, List[Any]]:
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_elo(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    In Elo we want numpy arrays for matchups and outcomes:
      matchups: int32 (N,2)  contains model ids for the competitors in a match
      outcomes: float64 (N,) contains 1.0, 0.5, or 0.0 representing win, tie, or loss for model_a
    """
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def preprocess_for_bt(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[Any], np.ndarray]:
    """
    In BT we only need the unique (matchup, outcome) sets along with the weights of how often they occur.
    """
    n_rows = len(df)
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    schedule[:, [0, 1]], models = get_matchups_models(df)
    schedule[df["winner"] == "model_a", 2] = 2
    schedule[df["winner"] == "model_b", 2] = 0
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    return matchups, outcomes, models, weights


def preprocess_for_style(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    add_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    apply_ratio = list(np.ones(len(style_elements) // 2))
    matchups, outcomes, models = preprocess_for_elo(df)
    n = matchups.shape[0]
    k = int(len(style_elements) / 2)

    def extract_style_feature(x: Dict[str, Any], feature: str) -> float:
        val = x[feature]
        if isinstance(val, (int, float)):
            return val
        else:
            return sum(val.values())

    style_vector = np.zeros(shape=(2 * k, n), dtype=np.int32)
    for idx, element in enumerate(style_elements):
        style_vector[idx, :] = df.conv_metadata.map(
            partial(extract_style_feature, feature=element)
        ).values
    style_vector = np.ascontiguousarray(style_vector)

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio_idx = np.flatnonzero(apply_ratio)
    style_diff[apply_ratio_idx] /= style_sum[apply_ratio_idx]

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return matchups, features, outcomes, models


def bt_loss_and_grad(
    ratings: np.ndarray,
    matchups: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    alpha: float = 1.0
) -> Tuple[float, np.ndarray]:
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    loss = -(
        (np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights
    ).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(
    matchups: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    n_models: int,
    alpha: float,
    tol: float = 1e-6
) -> np.ndarray:
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def scale_and_offset(
    ratings: np.ndarray,
    models: List[Any],
    scale: float,
    init_rating: float,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> np.ndarray:
    """
    Convert ratings from the natural scale to the Elo rating scale with an anchored baseline.
    """
    scaled_ratings = (ratings * scale) + init_rating
    if anchor_model_and_rating is not None:
        anchor_model, anchor_rating = anchor_model_and_rating
        baseline_idx = models.index(anchor_model)
        scaled_ratings += anchor_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings


def compute_bt(
    df: pd.DataFrame,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000,
    tol: float = 1e-6,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> pd.Series:
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base), tol)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles: pd.DataFrame,
    num_round: int,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    tol: float = 1e-6,
    num_cpu: Optional[int] = None,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
    offset: float = 0.0,
) -> pd.DataFrame:
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles), pvals=weights / weights.sum(), size=(num_round)
    )
    boot_weights = idxs.astype(np.float64) / len(battles)
    bt_fn = partial(
        fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(base), tol=tol
    )
    results = []
    for weights_ in boot_weights:
        results.append(bt_fn(weights_))
    ratings = np.array(results)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


DIFF_MASK: np.ndarray = np.array(
    [1.0, -1.0], dtype=np.float64
)  # create globally to not incur the instantiation cost in each call


def contextual_bt_loss_and_grad(
    params: np.ndarray,
    n_competitors: int,
    matchups: np.ndarray,
    features: np.ndarray,
    outcomes: np.ndarray,
    alpha: float = 1.0,
    reg: float = 1.0,
    half_reg: float = 0.5,
) -> Tuple[float, np.ndarray]:
    reg_loss = half_reg * np.inner(params, params)
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]
    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )
    error = outcomes - probs
    grad = reg * params
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)
    return loss, grad


def fit_contextual_bt(
    matchups: np.ndarray,
    features: np.ndarray,
    outcomes: np.ndarray,
    models: List[Any],
    idxs: Optional[np.ndarray] = None,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    tol: float = 1e-6,
) -> np.ndarray:
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0
    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]
    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    init_rating: float = 1000.0,
    scale: float = 400.0,
    tol: float = 1e-6,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> Tuple[pd.Series, np.ndarray]:
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)
    ratings_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = ratings_params[: len(models)]
    params = ratings_params[len(models):]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    scaled_ratings = pd.Series(scaled_ratings, index=models).sort_values(
        ascending=False
    )
    return scaled_ratings, params


def compute_bootstrap_style_control(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    num_round: int,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    init_rating: float = 1000.0,
    scale: float = 400.0,
    tol: float = 1e-6,
    num_cpu: Optional[int] = None,
    offset: float = 0.0,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)
    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    np.random.seed(0)
    boot_idxs = np.random.randint(
        low=0, high=matchups.shape[0], size=(num_round, matchups.shape[0])
    )
    results = []
    for idx in boot_idxs:
        results.append(contextual_bt_fn(idx))
    ratings_params = np.array(results)
    ratings = ratings_params[:, : len(models)]
    params = ratings_params[:, len(models):]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df_out = pd.DataFrame(scaled_ratings, columns=models)
    return df_out[df_out.median().sort_values(ascending=False).index], params


def get_model_order(
    battles: pd.DataFrame
) -> List[Any]:
    bt_ratings = compute_bt(battles)
    return list(bt_ratings.keys())


def bootstrap_winrates(
    data: pd.DataFrame
) -> Dict[Any, Dict[str, float]]:
    random.seed(42)
    data = data[data['winner'].isin(['model_a', 'model_b'])]
    data = data[data['model_a'] != data['model_b']]
    winner_a_count = data[data['winner'] == 'model_a'].pivot_table(index='model_a', columns='model_b', values='winner', aggfunc='count')
    winner_b_count = data[data['winner'] == 'model_b'].pivot_table(index='model_a', columns='model_b', values='winner', aggfunc='count')
    winner_count = winner_a_count + winner_b_count.T  # model_a > model_b
    bootstrap_winrates: Dict[Any, Dict[str, float]] = {}
    for model in winner_count.index:
        total_count = winner_count.loc[model].sum() + winner_count.T.loc[model].sum()
        win_count = winner_count.loc[model].sum()
        win_rate = win_count / total_count
        results = []
        for _ in range(1000):
            num_wins_sample = np.random.binomial(total_count, win_rate)
            results.append(num_wins_sample / total_count)
        bootstrap_winrates[model] = {
            "estimate": float(np.mean(results)),
            "ci_lower": float(np.mean(results) - np.percentile(results, 2.5)),
            "ci_upper": float(np.percentile(results, 97.5) - np.mean(results)),
        }
    return bootstrap_winrates


########################################
# Control Experiment Utils
########################################

def add_response_length_style(
    df: pd.DataFrame,
    style_name: str
) -> pd.DataFrame:
    if "conv_metadata" not in df.columns:
        df["conv_metadata"] = pd.Series([{} for _ in range(len(df))])
    for index in df.index:
        conv_metadata = df.loc[index, "conv_metadata"]
        conv_metadata[f"{style_name}_a"] = np.mean([
            len(message["content"].split())
            for message in df.loc[index, "messages_a"]
            if message["role"] == "assistant"
        ])
        conv_metadata[f"{style_name}_b"] = np.mean([
            len(message["content"].split())
            for message in df.loc[index, "messages_b"]
            if message["role"] == "assistant"
        ])
        df.at[index, "conv_metadata"] = conv_metadata
    return df

def get_num_urls(
    web_search_trace: List[dict]
) -> float:
    num_urls = 0
    for turn_trace in web_search_trace:
        num_urls += len(turn_trace)
    return num_urls / len(web_search_trace) if len(web_search_trace) > 0 else 0

def add_num_citations_style(
    df: pd.DataFrame,
    style_name: str
) -> pd.DataFrame:
    if "conv_metadata" not in df.columns:
        df["conv_metadata"] = pd.Series([{} for _ in range(len(df))])
    for index in df.index:
        conv_metadata = df.loc[index, "conv_metadata"]
        conv_metadata[f"{style_name}_a"] = get_num_urls(df.loc[index, f"system_a_metadata"]["web_search_trace"])
        conv_metadata[f"{style_name}_b"] = get_num_urls(df.loc[index, f"system_b_metadata"]["web_search_trace"])
        df.at[index, "conv_metadata"] = conv_metadata
    return df

US_NEWS_DOMAINS: List[str] = [
    "cnn.com",
    "apnews.com",
    "cnbc.com",
    "bloomberg.com",
    "economist.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "nbcnews.com",
    "abcnews.go.com",
    "usatoday.com",
    "npr.org",
    "latimes.com",
    "vox.com",
    "huffpost.com",
    "ft.com",
    "foxnews.com",
    "axios.com",
    "time.com",
    "buzzfeed.com",
    "cbsnews.com",
    "politico.co",
    "newsweek.com",
    "fortune.com",
    "theatlantic.com",
    "whattowatch.com",
    "scrippsnews.com",
    "investopedia.com",
    "yahoo.com",
    "breitbart.com",
    "washingtontimes.com",
    "dailycaller.com",
    "thefederalist.com",
    "townhall.com",
    "pjmedia.com",
    "westernjournal.com",
    "forbes.com"
]

FOREIGN_NEWS_DOMAINS: List[str] = [
    "reuters.com",
    "bbc.com",
    "aljazeera.com",
    "dw.com",
    "france24.com",
    "as.com",
    "elpais.com",
    "cbc.ca",
    "theglobeandmail.com",
    "smh.com.au",
    "abc.net.au",
    "japantimes.co.jp",
    "straitstimes.com",
    "hindustantimes.com",
    "thehindu.com",
    "economictimes.indiatimes.com",
    "indianexpress.com",
    "independent.co.uk",
    "theguardian.com",
    "cadenaser.com",
    "lemonde.fr",
    "vnexpress.net",
    "ndtv.com"
]

DOMAIN_CATEGORIES: List[str] = [
    "youtube",
    "gov_edu",
    "wiki",
    "us_news",
    "foreign_news",
    "social_media",
    "community_blog",
    "tech_coding",
    "academic_journal",
    "retail",
    "other"
]

def _get_domain_category(
    domain: str
) -> str:
    if "youtube.com" in domain:
        return "youtube"
    if any(x in domain for x in [".gov", ".edu", ".mil"]):
        return "gov_edu"
    if any(x in domain for x in ['wikipedia', 'wikihow', 'wikimedia']):
        return 'wiki'
    if any(x in domain for x in US_NEWS_DOMAINS):
        return 'us_news'
    elif any(x in domain for x in FOREIGN_NEWS_DOMAINS):
        return 'foreign_news'
    if any(x in domain for x in ['tiktok.com', 'facebook.com', 'instagram.com', 'x.com', 'twitter.com', 'linkedin.com', 'snapchat.com', 'pinterest.com']):
        return 'social_media'
    if any(x in domain for x in ['reddit.com', 'quora.com', 'blog', 'medium.com', 'wordpress.com', 'substack.com', 'tumblr.com']):
        return 'community_blog'
    if any(x in domain for x in ['github.com', 'gitlab.com', 'stackexchange.com', 'microsoft.com', 'dev.to', 'codecademy.com', 'stackoverflow.com']):
        return 'tech_coding'
    if any(x in domain for x in ['jstor.org', 'springer.com', 'sciencedirect.com', 'nature.com', 'arxiv.org', 'researchgate.com', 'biorxiv.org']):
        return 'academic_journal'
    if any(x in domain for x in ['amazon.com', 'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com', 'costco.com']):
        return 'retail'
    return 'other'

def _cites_domain_group(
    web_search_trace: List[Any],
    domain_group: str
) -> int:
    num_group_cited = 0
    for turn_trace in web_search_trace:
        for _, url in turn_trace:
            domain = url.split('/')[2]
            num_group_cited += (_get_domain_category(domain) == domain_group)
    return num_group_cited

def add_domain_style(
    df: pd.DataFrame,
    style_name: str
) -> pd.DataFrame:
    if "conv_metadata" not in df.columns:
        df["conv_metadata"] = pd.Series([{} for _ in range(len(df))])
    for index in df.index:
        conv_metadata = df.loc[index, "conv_metadata"]
        for domain_group in DOMAIN_CATEGORIES:
            num_cited_a = _cites_domain_group(df.loc[index, f"system_a_metadata"]["web_search_trace"], domain_group)
            num_cited_b = _cites_domain_group(df.loc[index, f"system_b_metadata"]["web_search_trace"], domain_group)
            conv_metadata[f"num_{style_name}_{domain_group}_a"] = num_cited_a
            conv_metadata[f"num_{style_name}_{domain_group}_b"] = num_cited_b
            conv_metadata[f"{style_name}_{domain_group}_a"] = 1 if num_cited_a > 0 else 0
            conv_metadata[f"{style_name}_{domain_group}_b"] = 1 if num_cited_b > 0 else 0
        df.at[index, "conv_metadata"] = conv_metadata
    return df

def add_cit_misattribution_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "conv_metadata" not in df.columns:
        df["conv_metadata"] = pd.Series([{} for _ in range(len(df))])
    for index in df.index:
        conv_metadata = df.loc[index, "conv_metadata"]
        conv_metadata["support_count_a"] = df.loc[index, "support_count_a"] / df.loc[index, "turn"]
        conv_metadata["support_count_b"] = df.loc[index, "support_count_b"] / df.loc[index, "turn"]
        conv_metadata["irrelevant_count_a"] = df.loc[index, "irrelevant_count_a"] / df.loc[index, "turn"]
        conv_metadata["irrelevant_count_b"] = df.loc[index, "irrelevant_count_b"] / df.loc[index, "turn"]
        conv_metadata["contradict_count_a"] = df.loc[index, "contradict_count_a"] / df.loc[index, "turn"]
        conv_metadata["contradict_count_b"] = df.loc[index, "contradict_count_b"] / df.loc[index, "turn"]
        df.at[index, "conv_metadata"] = conv_metadata
    return df
