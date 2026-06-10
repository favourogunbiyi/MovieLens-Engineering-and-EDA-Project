# MovieLens-Engineering-and-EDA-Project
Wrestling with 100k raw MovieLens ratings to find out what actually makes a movie a cult classic. Cleared 52k missing tags and engineered 8 features from scratch.
Raw ratings tell you what score a film got. This project engineers the eight features that explain *why* — and surfaces the patterns a recommendation engine actually needs.

# MovieLens Feature Engineering & Exploratory Data Analysis

## Tools and Concept

Python · Pandas · NumPy · Matplotlib · Seaborn · Power Query · Feature Engineering · EDA · Data Cleaning · Recommendation Systems

## At a Glance

| | |
|---|---|
| **Dataset** | 100,836 ratings · 9,742 movies · 610 users · 4 source files |
| **Tools** | Python (Pandas, NumPy, Matplotlib, Seaborn) · Power Query |
| **Deliverables** | 8 engineered features · 6 EDA insights · cleaned 100,823-row dataset · written report |
| **Key finding** | 48.2% of all ratings are 4.0+, yet Film-Noir averages 0.68 stars higher than Horror — the distribution hides more than it reveals |

## The Problem

Streaming platforms fail not for lack of data but for lack of structure. A flat ratings table tells you scores. It doesn't tell you that 1940s films average 3.87 — or that Pulp Fiction has 181 user tags while most titles have none. This project engineers that context into the MovieLens dataset, turning four raw CSVs into a recommendation-ready feature table.

## The Dataset

| File | Contents | Rows |
|---|---|---|
| `ratings.csv` | userId, movieId, rating (0.5–5.0), Unix timestamp | 100,836 |
| `movies.csv` | movieId, title (year embedded), pipe-separated genres | 9,742 |
| `tags.csv` | userId, movieId, free-text user tag, timestamp | 3,683 |
| `links.csv` | movieId, IMDb ID, TMDb ID | 9,742 |

**What made it messy:**

- 52,549 tag values missing after the merge — 8,170 of 9,742 movies had zero user tags
- 13 rows with no TMDb ID after the join — no recovery path
- Release years buried inside title strings (`"Toy Story (1995)"`) — no dedicated column
- Timestamps stored as raw Unix integers (`964982703`), not readable dates
- Genres pipe-concatenated in a single string (`"Adventure|Animation|Children|Comedy|Fantasy"`)

## Part A — Data Cleaning

| Problem Found | Action Taken | Impact |
|---|---|---|
| 52,549 NULL tag values | Filled with `'No Tag'` — kept all rating rows intact | Zero data loss on ratings |
| 13 rows, missing TMDb ID | Dropped — no external identifier, no recovery | 13 rows excluded; documented |
| Unix timestamps unreadable | Converted with `pd.to_datetime(unit='s')` | `year` and `month` columns usable |
| Duplicate check | `df.duplicated().sum()` → returned 0 | Confirmed clean, no action needed |
| Tags many-to-many before merge | Grouped tags by movieId before joining | No row multiplication on merge |

**After cleaning: 100,823 rows · zero nulls · zero duplicates**

## Part B — Feature Engineering

| Feature | Built With | Purpose |
|---|---|---|
| `release_year` | Regex on title string | Enables decade grouping and temporal analysis |
| `num_genres` | `len(genres.split('\|'))` | Tests whether genre diversity lifts ratings |
| `avg_rating` | `groupby('movieId')['rating'].mean()` | Core quality signal per film |
| `user_rating_freq` | `groupby('userId')['rating'].count()` | Flags power users who may skew aggregates |
| `movie_age` | `2025 - release_year` | Measures nostalgia vs recency bias |
| `age_group` | `pd.cut()` into 7 brackets | Era-cohort comparisons |
| `num_tags` | `groupby('movieId')['tag'].count()` | Engagement signal — distinct from raw popularity |
| `year`, `month` | `.dt.year` / `.dt.month` | Time-series and seasonality analysis |

## Part C — Six Insights

### 1. Ratings are positively skewed — and the compression is a problem · [View →](./Stage1_Feature_Engineering.ipynb)

**48.2% of all 100,836 ratings are 4.0 or above. Only 5.9% fall below 2.0.** The mean sits at 3.50. This positive bias compresses meaningful differences between films — a recommender that uses raw ratings alone will struggle to separate "great" from "good."

### 2. 1940s films average 3.87 — nostalgia is measurable, not just a feeling · [View →](./Stage1_Feature_Engineering.ipynb)

**Ratings peak in the 1940s (3.87) and decline steadily through the 1990s (3.44) before a modest recovery in the 2010s (3.49).** The likely cause: survival bias. Bad films from the 1940s have no ratings because no one kept watching them. `release_year` and `decade` are real signals for a recommendation model.

### 3. Multi-genre films rate consistently higher · [View →](./Stage1_Feature_Engineering.ipynb)

**Films tagged with 10 genres sit at the top of the average-rating chart; single-genre films sit lowest.** Multi-genre titles attract broader audiences, which smooth out extremely low ratings. `num_genres` is worth including in a hybrid recommender as a content-breadth signal.

### 4. Films aged 61–80 years rate highest overall (~3.80 average) · [View →](./Stage1_Feature_Engineering.ipynb)

**The sweet spot is mid-century cinema.** Films 0–20 years old average ~3.50; films 61–80 years old average ~3.80. The very oldest (120+ years) drop back to ~3.25 — thin catalogue, obscure titles. `movie_age` and `age_group` together capture this curve for personalisation.

---

### 5. Film-Noir (~3.93) leads all genres; Horror (~3.25) trails by 0.68 stars · [View →](./Stage1_Feature_Engineering.ipynb)

**Niche genres outperform mass-market ones — not because they're better, but because only engaged audiences seek them.** Comedy (~3.38) and Horror attract everyone, including people who don't like them, dragging the average down. A recommender that treats all genres as equal signals will systematically misrank both ends.

### 6. Pulp Fiction has 181 tags — 3.3× more than any other title · [View →](./Stage1_Feature_Engineering.ipynb)

**The top 10 most-tagged films are almost all cult classics: Fight Club (54), 2001: A Space Odyssey (41), Donnie Darko (29).** These are not the most-rated films — they are the most discussed. `num_tags` measures cultural engagement, which is a distinct and complementary signal to `avg_rating`.

## Part D — Recommendation System Output · [Read Report →](./MovieLens_Stage1_Final_Report.pdf)

**3 findings. 4 feature recommendations.**

**Finding 1 — Ratings are a blunt instrument.** With 48.2% of ratings at 4.0+, raw scores need to be paired with count-based and engagement signals to be meaningful.

**Finding 2 — Time matters.** Decade and movie age both correlate with ratings in consistent, interpretable ways. Ignoring when a film was made discards a real signal.

**Finding 3 — Engagement ≠ popularity.** Tag count and rating count measure different things. Conflating them misses an entire layer of user preference.

**Feature recommendations for the model:** use `movie_age` to surface nostalgia-appropriate picks · use `num_genres` to match breadth-seeking users with diverse titles · use `num_tags` alongside `avg_rating` to elevate culturally engaged films even when their raw score is mid-range · use `user_rating_freq` to down-weight power users in collaborative filtering so 30 active raters don't dominate everyone else's recommendations.

## What This Analysis Doesn't Do

- **No model was built.** Features are engineered and validated as signals — not tested in a live recommendation system with train/test evaluation.
- **Tag text was never analysed, only counted.** The words inside tags ("mindfuck," "twist ending") are richer than any count. A TF-IDF pass is the most obvious next step.
- **52,549 missing tags are structural, not accidental.** 84% of movies have zero tags. `num_tags` carries no signal for those titles — any model using it needs to handle the sparsity.
- **The nostalgia effect wasn't statistically tested.** Decade-based rating differences are real; whether the cause is survival bias, volume, or something else is unconfirmed.
- **No user demographics.** 610 users, no age, location, or external context. Cold-start personalisation isn't possible with this dataset alone.

## Files

| File | Contents |
|---|---|
| [Stage1_Feature_Engineering.ipynb](./Stage1_Feature_Engineering.ipynb) | Full notebook — cleaning, features, EDA, all visualisations |
| [cleaned_movielens.csv](./cleaned_movielens.csv) | Output dataset — 100,823 rows, 18 columns |
| [MovieLens_Stage1_Final_Report.pdf](./MovieLens_Stage1_Final_Report.pdf) | Written report — features, insights, recommendations |
| `movies.csv` · `ratings.csv` · `tags.csv` · `links.csv` | Raw source files |

## Context

**HNG Internship (HNGi13) · Stage 1 · Data Analytics Track**
Remote · self-directed · deadline: 22nd October 2025

*Currently building: Star Schema design · pipeline architecture*

[← Back to portfolio](#)
