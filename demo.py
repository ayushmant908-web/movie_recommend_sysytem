
import os
import pickle
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request


# ---------------- CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


# ---------------- LOAD DATA ----------------

movies = pd.read_pickle(
    os.path.join(BASE_DIR, "movie_list.pkl")
).reset_index(drop=True)

if "title" not in movies.columns:
    raise ValueError("movie_list.pkl must contain a 'title' column.")

with open(
    os.path.join(BASE_DIR, "top5_simaliarity.pkl"), "rb"
) as file:
    top_movies = pickle.load(file)


# ---------------- FETCH TMDB POSTER ----------------



# ---------------- FETCH TMDB POSTER ----------------

@lru_cache(maxsize=2048)
def fetch_poster(movie_id, movie_title=""):

    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY is missing in .env")
        return None

    headers = {
        "accept": "application/json"
    }

    def get_poster(data):
        poster_path = data.get("poster_path")

        if poster_path:
            return f"{POSTER_BASE_URL}{poster_path}"

        return None

    # Step 1: Try using TMDB movie ID
    if movie_id is not None and str(movie_id).strip():

        try:
            if not pd.isna(movie_id):
                numeric_id = int(float(movie_id))

                response = requests.get(
                    f"https://api.themoviedb.org/3/movie/{numeric_id}",
                    params={
                        "api_key": TMDB_API_KEY,
                        "language": "en-US"
                    },
                    headers=headers,
                    timeout=10
                )

                print(
                    f"ID lookup: {movie_title} | "
                    f"ID: {numeric_id} | "
                    f"Status: {response.status_code}"
                )

                if response.status_code == 200:
                    poster_url = get_poster(response.json())

                    if poster_url:
                        return poster_url

        except (ValueError, TypeError, OverflowError):
            print("Invalid TMDB ID:", movie_id)

        except requests.RequestException as e:
            print("TMDB ID request failed:", e)

    # Step 2: If ID lookup fails, search by movie title
    if not movie_title or not str(movie_title).strip():
        return None

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": str(movie_title).strip(),
                "include_adult": "false",
                "language": "en-US",
                "page": 1
            },
            headers=headers,
            timeout=10
        )

        print(
            f"Title search: {movie_title} | "
            f"Status: {response.status_code}"
        )

        if response.status_code != 200:
            print("TMDB search error:", response.text[:200])
            return None

        results = response.json().get("results", [])

        if not results:
            print("No TMDB result for:", movie_title)
            return None

        # Prefer an exact title match
        target = str(movie_title).strip().casefold()

        exact_matches = [
            result for result in results
            if result.get("title", "").strip().casefold() == target
            and result.get("poster_path")
        ]

        if exact_matches:
            return get_poster(exact_matches[0])

        # Otherwise, use the first result that has a poster
        for result in results:
            poster_url = get_poster(result)

            if poster_url:
                return poster_url

        print("No poster available for:", movie_title)
        return None

    except requests.RequestException as e:
        print("TMDB title search failed:", e)
        return None


# ---------------- PREPARE MOVIE ----------------

def prepare_movie(row):

    movie = row.to_dict()

    # Support common movie ID column names
    movie_id = None

    for column in (
        "movie_id", "tmdb_id", "tmdbId", "movieId", "id"
    ):
        value = movie.get(column)

        if value is not None and not pd.isna(value):
            movie_id = value
            break

    movie_title = str(movie.get("title", "")).strip()

    movie["poster_url"] = fetch_poster(
        movie_id,
        movie_title
    )

    return movie


# ---------------- HOME: TOP 50 ----------------

@app.route("/")
def home():

    home_movies = [
        prepare_movie(row)
        for _, row in movies.head(50).iterrows()
    ]

    return render_template(
        "index.html",
        active_tab="home",
        home_movies=home_movies
    )


# ---------------- RECOMMEND TOP 5 ----------------

@app.route("/recommend", methods=["GET", "POST"])
def recommendation():

    movie_titles = (
        movies["title"].dropna().astype(str).tolist()
    )

    recommendations = []
    selected_movie = ""
    error = None

    if request.method == "POST":

        selected_movie = request.form.get(
            "movie", ""
        ).strip()

        matches = movies.index[
            movies["title"].astype(str) == selected_movie
        ].tolist()

        if not matches:
            error = "Movie not found. Please select another movie."

        else:
            movie_index = matches[0]

            indices = top_movies.get(
                movie_index,
                top_movies.get(str(movie_index), [])
            )

            for idx in indices:

                if len(recommendations) >= 5:
                    break

                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    continue

                if 0 <= idx < len(movies) and idx != movie_index:
                    row = movies.iloc[idx]
                    recommendations.append(prepare_movie(row))

            if not recommendations:
                error = "No recommendations found for this movie."

    return render_template(
        "index.html",
        active_tab="recommend",
        movie_titles=movie_titles,
        recommendations=recommendations,
        selected_movie=selected_movie,
        error=error
    )


# ---------------- CONTACT ----------------

@app.route("/contact", methods=["GET", "POST"])
def contact():

    success = False
    error = None

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            error = "Please fill in all fields."

        elif (
            "@" not in email
            or "." not in email.split("@")[-1]
        ):
            error = "Please enter a valid email address."

        else:
            # Demo only: the message is not saved or emailed.
            success = True

    return render_template(
        "index.html",
        active_tab="contact",
        success=success,
        error=error
    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    print("TMDB API key loaded:", bool(TMDB_API_KEY))
    app.run(debug=True)