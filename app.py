from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os


# =========================
# Config (Windows paths)
# =========================
import os

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "Chill_vs_Duke.joblib")
LINKS_CSV_PATH = os.getenv("LINKS_CSV_PATH", "links.csv")

# =========================
# Helper functions
# =========================
def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize finite values only (ignores NaN/inf). Non-finite -> 0."""
    out = np.zeros_like(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    v = arr[mask]
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-8:
        out[mask] = 0.0
        return out
    out[mask] = (arr[mask] - vmin) / (vmax - vmin)
    return out


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores (descending). Assumes scores is 1D."""
    n = int(scores.shape[0])
    k = min(int(k), n)
    if k <= 0:
        return np.array([], dtype=np.int32)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int32)


def _normalize_weights(wc: float, wf: float, cf_ready: bool) -> tuple[float, float]:
    """
    Robust weights:
      - If CF not ready -> wf=0, wc=1
      - Clip negatives to 0
      - Normalize so wc+wf=1
    """
    if not cf_ready:
        return 1.0, 0.0
    wc = max(0.0, float(wc))
    wf = max(0.0, float(wf))
    s = wc + wf
    if s <= 1e-8:
        return 0.5, 0.5
    return wc / s, wf / s


# =========================
# Core recommender class
# =========================
class ChillVsDukeRecommender:
    """
    Loads Chill_vs_Duke.joblib exported from notebook.
    Expected keys:
      - model_name (optional)
      - movie_ids, titles, genres (genres optional)
      - id2row (optional)
      - X  (content embeddings; here Z_norm)
      - w_content, w_cf
      - ITEM_EMB_CF (optional), mid2col_cf (optional)
      - cf_item_rows_valid, cf_valid (optional)
    """

    def __init__(self, artifact_path: str, links_csv_path: str):
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        if not os.path.exists(links_csv_path):
            raise FileNotFoundError(f"links.csv not found: {links_csv_path}")

        self.A = joblib.load(artifact_path)

        # --- Load model universe ---
        self.model_name = str(self.A.get("model_name", "Chill_vs_Duke"))

        self.movie_ids: np.ndarray = np.asarray(self.A["movie_ids"], dtype=np.int64)
        self.titles: list[str] = list(self.A["titles"])
        self.genres: list[str] = list(self.A.get("genres", [""] * len(self.movie_ids)))

        # Content embeddings: X = Z_norm (already normalized)
        self.X: np.ndarray = np.asarray(self.A["X"], dtype=np.float32)
        if self.X.ndim != 2 or self.X.shape[0] != len(self.movie_ids):
            raise ValueError(f"Bad X shape: {self.X.shape} vs movies {len(self.movie_ids)}")

        # id->row mapping
        if "id2row" in self.A and isinstance(self.A["id2row"], dict):
            self.id2row = {int(k): int(v) for k, v in self.A["id2row"].items()}
        else:
            self.id2row = {int(mid): i for i, mid in enumerate(self.movie_ids)}

        # Default weights
        self.w_content_default = float(self.A.get("w_content", 0.6))
        self.w_cf_default = float(self.A.get("w_cf", 0.4))

        # --- CF optional ---
        self.ITEM_EMB_CF = self.A.get("ITEM_EMB_CF", None)
        if self.ITEM_EMB_CF is not None:
            self.ITEM_EMB_CF = np.asarray(self.ITEM_EMB_CF, dtype=np.float32)

        self.mid2col_cf = self.A.get("mid2col_cf", None)

        self.cf_item_rows_valid = self.A.get("cf_item_rows_valid", None)
        if self.cf_item_rows_valid is not None:
            self.cf_item_rows_valid = np.asarray(self.cf_item_rows_valid, dtype=np.int32)

        self.cf_valid = self.A.get("cf_valid", None)
        if self.cf_valid is not None:
            self.cf_valid = np.asarray(self.cf_valid, dtype=bool)

        self.cf_ready = (
            self.ITEM_EMB_CF is not None
            and isinstance(self.mid2col_cf, dict)
            and self.cf_item_rows_valid is not None
            and self.cf_valid is not None
        )

        # --- Load links.csv for TMDb ↔ MovieLens mapping ---
        links = pd.read_csv(links_csv_path)
        links["tmdbId"] = links["tmdbId"].astype("Int64")  # nullable

        links_ok = links.dropna(subset=["tmdbId"]).copy()
        links_ok["tmdbId"] = links_ok["tmdbId"].astype(int)
        links_ok["movieId"] = links_ok["movieId"].astype(int)

        universe = set(int(x) for x in self.movie_ids)
        links_ok = links_ok[links_ok["movieId"].isin(universe)]

        self.tmdb_to_ml = dict(zip(links_ok["tmdbId"], links_ok["movieId"]))
        self.ml_to_tmdb = dict(zip(links_ok["movieId"], links_ok["tmdbId"]))
        self.tmdb_coverage = len(self.ml_to_tmdb)

    def recommend_by_movielens_movieId(
        self,
        seed_movie_id: int,
        k: int = 10,
        exclude_movie_ids: list[int] | None = None,
        w_content: float | None = None,
        w_cf: float | None = None,
    ) -> list[dict]:
        seed_movie_id = int(seed_movie_id)
        if seed_movie_id not in self.id2row:
            raise KeyError("Unknown MovieLens movieId")

        exclude_set = set(int(x) for x in (exclude_movie_ids or []))
        exclude_set.add(seed_movie_id)

        i = self.id2row[seed_movie_id]

        # ---- Content similarity ----
        # X is Z_norm so dot == cosine
        v = self.X[i]
        content_scores = (self.X @ v).astype(np.float32)

        content_scores[i] = np.nan
        for mid in exclude_set:
            r = self.id2row.get(mid)
            if r is not None:
                content_scores[r] = np.nan

        # ---- CF similarity mapped to full universe (NaN if missing) ----
        cf_scores_full = np.full(self.X.shape[0], np.nan, dtype=np.float32)
        if self.cf_ready and seed_movie_id in self.mid2col_cf:
            j = int(self.mid2col_cf[seed_movie_id])
            v_cf = self.ITEM_EMB_CF[j]
            sim_cf_items = (self.ITEM_EMB_CF @ v_cf).astype(np.float32)

            cf_scores_full[self.cf_item_rows_valid] = sim_cf_items[self.cf_valid]
            cf_scores_full[i] = np.nan

            for mid in exclude_set:
                r = self.id2row.get(mid)
                if r is not None:
                    cf_scores_full[r] = np.nan

        # ---- Normalize + Hybrid ----
        wc = self.w_content_default if w_content is None else float(w_content)
        wf = self.w_cf_default if w_cf is None else float(w_cf)
        wc, wf = _normalize_weights(wc, wf, cf_ready=self.cf_ready)

        content_norm = _minmax_norm(content_scores)
        cf_norm = _minmax_norm(cf_scores_full)

        hybrid = wc * content_norm + wf * cf_norm

        # hard exclude
        hybrid[i] = -np.inf
        for mid in exclude_set:
            r = self.id2row.get(mid)
            if r is not None:
                hybrid[r] = -np.inf

        top_idx = _topk_indices(hybrid, k)
        recs: list[dict] = []
        for idx in top_idx:
            recs.append(
                {
                    "movie_id": int(self.movie_ids[idx]),
                    "title": self.titles[idx],
                    "genres": self.genres[idx] if idx < len(self.genres) else "",
                    "hybrid_score": float(hybrid[idx]),
                    "content_norm": float(content_norm[idx]),
                    "cf_norm": float(cf_norm[idx]),
                    "cf_available": bool(np.isfinite(cf_scores_full[idx])),
                }
            )
        return recs

    def recommend_by_tmdb_id(
        self,
        seed_tmdb_id: int,
        k: int = 10,
        exclude_tmdb_ids: list[int] | None = None,
        w_content: float | None = None,
        w_cf: float | None = None,
    ) -> dict:
        seed_tmdb_id = int(seed_tmdb_id)
        seed_ml = self.tmdb_to_ml.get(seed_tmdb_id)
        if seed_ml is None:
            raise KeyError("TMDb id not found in links.csv or not in model universe")

        exclude_tmdb_ids = exclude_tmdb_ids or []
        exclude_ml: list[int] = []
        for t in exclude_tmdb_ids + [seed_tmdb_id]:
            ml = self.tmdb_to_ml.get(int(t))
            if ml is not None:
                exclude_ml.append(int(ml))

        oversample = min(max(k * 5, 50), 300)
        recs_ml = self.recommend_by_movielens_movieId(
            seed_movie_id=int(seed_ml),
            k=oversample,
            exclude_movie_ids=exclude_ml,
            w_content=w_content,
            w_cf=w_cf,
        )

        recs_tmdb: list[dict] = []
        for r in recs_ml:
            tmdb = self.ml_to_tmdb.get(int(r["movie_id"]))
            if tmdb is None:
                continue
            recs_tmdb.append(
                {
                    "tmdb_id": int(tmdb),
                    "score": float(r["hybrid_score"]),
                    "movielens_movie_id": int(r["movie_id"]),
                    "title": r["title"],
                    "genres": r.get("genres", ""),
                    "content_norm": float(r["content_norm"]),
                    "cf_norm": float(r["cf_norm"]),
                    "cf_available": bool(r.get("cf_available", False)),
                }
            )
            if len(recs_tmdb) >= k:
                break

        return {
            "model": self.model_name,
            "seed_tmdb_id": seed_tmdb_id,
            "seed_movielens_movie_id": int(seed_ml),
            "k": int(k),
            "tmdb_coverage_in_model_universe": int(self.tmdb_coverage),
            "recommendations": recs_tmdb,
        }


# =========================
# FastAPI app (lifespan)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.reco = ChillVsDukeRecommender(ARTIFACT_PATH, LINKS_CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize recommender: {e}") from e
    yield


app = FastAPI(
    title="Chill_vs_Duke Hybrid Movie Recommender API",
    version="2.0",
    lifespan=lifespan,
)


def get_reco(app: FastAPI) -> ChillVsDukeRecommender:
    reco = getattr(app.state, "reco", None)
    if reco is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    return reco


@app.get("/health")
def health():
    reco = get_reco(app)
    return {
        "status": "ok",
        "model": reco.model_name,
        "movies": int(len(reco.movie_ids)),
        "cf_ready": bool(reco.cf_ready),
        "tmdb_coverage_in_model_universe": int(reco.tmdb_coverage),
    }


class RecommendTMDbRequest(BaseModel):
    seed_tmdb_id: int
    k: int = Field(default=10, ge=1, le=50)
    exclude_tmdb_ids: list[int] = Field(default_factory=list)
    w_content: float | None = None
    w_cf: float | None = None


@app.post("/recommend_tmdb")
def recommend_tmdb(req: RecommendTMDbRequest):
    reco = get_reco(app)
    try:
        return reco.recommend_by_tmdb_id(
            seed_tmdb_id=req.seed_tmdb_id,
            k=req.k,
            exclude_tmdb_ids=req.exclude_tmdb_ids,
            w_content=req.w_content,
            w_cf=req.w_cf,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
