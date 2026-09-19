#!/usr/bin/env python3
"""
music_scraper.py — Music metadata collector for a recommendation system.

Sources
-------
  * YouTube Music  (via the `ytmusicapi` unofficial client)

What it collects
----------------
  1. Track metadata      -> content-based features (title, artists, genre, duration, year)
  2. Playlist membership -> co-occurrence signal (the strongest free CF signal you can get)
  3. "Related track" edges -> item-item graph, ready for random-walk / ALS style models

Install
-------
    pip install requests beautifulsoup4 lxml ytmusicapi rapidfuzz tenacity

Usage
-----
    # Seed from charts, then expand 2 hops through related tracks
    python music_scraper.py crawl --seed-charts --depth 2 --max-tracks 5000

    # Seed from your own search terms
    python music_scraper.py crawl \
        --queries "Sơn Tùng M-TP" "Hoàng Thùy Linh" "indie việt" --max-tracks 800

    # Emit training files for the recommender
    python music_scraper.py export --out ./dataset

Notes
-----
  * Scrape responsibly: this script rate-limits every source, honours robots.txt by
    default (--ignore-robots to override), and caches responses so re-runs are cheap.
    Check each site's Terms of Service before collecting at scale, and prefer official
    APIs (YouTube Data API v3) where your use case allows.
  * Private endpoints change. Every selector / API key lives in the CONFIG block below
    so you can patch one place when a site updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from ytmusicapi import YTMusic
except ImportError:
    YTMusic = None


# --------------------------------------------------------------------------------------
# CONFIG — patch here when a site changes
# --------------------------------------------------------------------------------------

LOG = logging.getLogger("music_scraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.3 Safari/605.1.15",
]

class YoutubeConfig:
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    STATS_URL = "https://www.googleapis.com/youtube/v3/videos"

YTM_CHART_COUNTRIES = ["VN", "US", "ZZ"]  # ZZ == global


# --------------------------------------------------------------------------------------
# Unified data model
# --------------------------------------------------------------------------------------


@dataclass
class Track:
    source: str
    source_id: str
    title: str
    artists: list[str] = field(default_factory=list)
    album: str | None = None
    duration_sec: int | None = None
    genres: list[str] = field(default_factory=list)
    release_date: str | None = None
    play_count: int | None = None
    like_count: int | None = None
    thumbnail: str | None = None
    url: str | None = None
    lyrics_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def title_norm(self) -> str:
        return normalize_text(self.title)

    @property
    def artists_norm(self) -> str:
        return normalize_text(", ".join(sorted(self.artists)))


@dataclass
class Playlist:
    source: str
    source_id: str
    title: str
    owner: str | None = None
    track_uids: list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"


@dataclass
class Edge:
    src_uid: str
    dst_uid: str
    kind: str
    weight: float = 1.0


# --------------------------------------------------------------------------------------
# Text normalisation (Vietnamese-aware)
# --------------------------------------------------------------------------------------

_PAREN_NOISE = re.compile(
    r"\((?:[^)]*(?:official|mv|lyric|audio|remix|beat|karaoke|cover|ver\.?|version|"
    r"live|explicit|feat\.?|ft\.?)[^)]*)\)",
    re.I,
)
_BRACKET_NOISE = re.compile(r"\[[^\]]*\]")
_FEAT = re.compile(r"\s*(?:feat\.?|ft\.?|cùng với|with)\s+.*$", re.I)
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    """'Nước Ngoài' -> 'Nuoc Ngoai'. Keeps đ/Đ handled explicitly."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = _PAREN_NOISE.sub(" ", s)
    s = _BRACKET_NOISE.sub(" ", s)
    s = _FEAT.sub(" ", s)
    s = strip_diacritics(s)
    s = _NON_WORD.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def parse_duration(value: Any) -> int | None:
    """Accepts 213, '213', '3:33', '1:02:03'."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "tr": 1_000_000, "n": 1_000}


def parse_count(value: Any) -> int | None:
    """Handles '1.2M', '1,2M', '1,234,567 lượt nghe', '1.234.567', 2_500_000."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()

    # Suffixed short form: the separator is a decimal point, e.g. 1.2M / 1,2 tr
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(tr|[KMBN])\b", text, re.I)
    if m:
        number = float(m.group(1).replace(",", "."))
        return int(number * _MULT[m.group(2).lower()])

    # Plain integer with thousand separators (either convention).
    m = re.search(r"\d[\d.,\s]*", text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(0))
    return int(digits) if digits else None


# --------------------------------------------------------------------------------------
# HTTP plumbing: rate limiting, retries, caching, robots.txt
# --------------------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe token spacing with jitter, per source."""

    def __init__(self, min_interval: float = 1.0, jitter: float = 0.4):
        self.min_interval = min_interval
        self.jitter = jitter
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.min_interval + random.uniform(0, self.jitter)


class RobotsGate:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cache: dict[str, robotparser.RobotFileParser] = {}
        self._lock = threading.Lock()

    def allowed(self, url: str, ua: str = "*") -> bool:
        if not self.enabled:
            return True
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            rp = self._cache.get(root)
            if rp is None:
                rp = robotparser.RobotFileParser()
                rp.set_url(urljoin(root, "/robots.txt"))
                try:
                    rp.read()
                except Exception as exc:  # network error -> fail open but warn
                    LOG.warning("robots.txt unreachable for %s (%s); allowing", root, exc)
                    rp = None
                self._cache[root] = rp  # type: ignore[assignment]
        if rp is None:
            return True
        return rp.can_fetch(ua, url)


class HttpCache:
    """Tiny on-disk response cache so re-crawls don't re-hit the sites."""

    def __init__(self, directory: Path, ttl_sec: int = 7 * 24 * 3600, enabled: bool = True):
        self.dir = directory
        self.ttl = ttl_sec
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / (hashlib.sha1(key.encode()).hexdigest() + ".cache")

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists() or time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        try:
            self._path(key).write_text(value, encoding="utf-8")
        except OSError as exc:
            LOG.debug("cache write failed: %s", exc)


class BaseScraper:
    name = "base"

    def __init__(self, cache: HttpCache, robots: RobotsGate, min_interval: float = 1.0):
        self.cache = cache
        self.robots = robots
        self.limiter = RateLimiter(min_interval)
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update(
            {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            }
        )
        return s

    def fetch(self, url: str, *, params: dict | None = None, as_json: bool = False,
              cache_key: str | None = None, headers: dict | None = None) -> Any:
        key = cache_key or f"{url}?{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        cached = self.cache.get(key)
        if cached is not None:
            return json.loads(cached) if as_json else cached

        if not self.robots.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")

        self.limiter.wait()
        resp = self.session.get(url, params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        text = resp.text
        self.cache.put(key, text)
        return resp.json() if as_json else text

    # Subclasses implement whichever of these make sense.
    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        raise NotImplementedError

    def chart_tracks(self, limit: int = 100) -> list[Track]:
        raise NotImplementedError

    def related_tracks(self, track: Track, limit: int = 20) -> list[Track]:
        return []

    def playlists_for(self, track: Track, limit: int = 5) -> list[Playlist]:
        return []


# --------------------------------------------------------------------------------------
# YouTube Music
# --------------------------------------------------------------------------------------


class YouTubeMusicScraper(BaseScraper):
    name = "youtube_music"

    def __init__(self, cache: HttpCache, robots: RobotsGate, auth_file: str | None = None):
        super().__init__(cache, robots, min_interval=0.8)
        if YTMusic is None:
            raise RuntimeError("pip install ytmusicapi to use the YouTube Music source")
        # auth_file (browser.json from `ytmusicapi browser`) unlocks library/history
        # endpoints — great for building real user-item interaction data.
        if auth_file and Path(auth_file).exists():
            LOG.info("[ytm] using authenticated session: %s", auth_file)
            self.client = YTMusic(auth_file)
        else:
            LOG.warning(
                "[ytm] no browser authentication found; "
                "top-song charts may be unavailable"
            )
            self.client = YTMusic()

        self.youtube_api_key = YoutubeConfig.API_KEY

    # ------------------------------------------------------------------
    # YouTube statistics
    # ------------------------------------------------------------------
    def _youtube_stats(self, video_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        """
        Fetch authoritative public YouTube statistics.

        Returns:
            {
                "VIDEO_ID": {
                    "view_count": ...,
                    "like_count": ...
                }
            }

        YouTube Data API allows up to 50 IDs per videos.list request.
        """
        if not self.youtube_api_key or not video_ids:
            return {}

        stats: dict[str, dict[str, int]] = {}

        # videos.list accepts at most 50 IDs per request.
        for i in range(0, len(video_ids), 50):
            chunk = list(dict.fromkeys(video_ids[i:i + 50]))

            try:
                self.limiter.wait()

                response = self.session.get(
                    YoutubeConfig.STATS_URL,
                    params={
                        "part": "statistics",
                        "id": ",".join(chunk),
                        "key": self.youtube_api_key,
                    },
                    timeout=25,
                )
                response.raise_for_status()
                data = response.json()

            except requests.RequestException as exc:
                LOG.warning(
                    "[ytm] YouTube statistics request failed: %s",
                    exc,
                )
                continue

            for item in data.get("items", []):
                video_id = item.get("id")
                statistics = item.get("statistics") or {}

                if not video_id:
                    continue

                stats[video_id] = {
                    "view_count": parse_count(statistics.get("viewCount")),
                    "like_count": parse_count(statistics.get("likeCount")),
                }

        return stats

    def _enrich_statistics(self, tracks: list[Track]) -> list[Track]:
        """
        Fill view_count / like_count without overwriting existing values.

        If no YouTube Data API key is configured, fall back to get_song()
        for viewCount where possible.
        """
        if not tracks:
            return tracks

        # Preferred path: official YouTube Data API.
        if self.youtube_api_key:
            stats = self._youtube_stats([t.source_id for t in tracks])

            for track in tracks:
                s = stats.get(track.source_id)
                if not s:
                    continue

                if s.get("view_count") is not None:
                    track.play_count = s["view_count"]

                if s.get("like_count") is not None:
                    track.like_count = s["like_count"]

            return tracks

        # Fallback: ytmusicapi get_song() provides videoDetails.viewCount,
        # but not a reliable public like count.
        for track in tracks:
            try:
                self.limiter.wait()
                song = self.client.get_song(track.source_id)

                video_details = song.get("videoDetails") or {}

                view_count = parse_count(
                    video_details.get("viewCount")
                )

                if view_count is not None:
                    track.play_count = view_count

            except Exception as exc:
                LOG.debug(
                    "[ytm] statistics lookup failed for %s: %s",
                    track.source_id,
                    exc,
                )

        return tracks

    # -- mapping -----------------------------------------------------------------
    def _to_track(self, item: dict) -> Track | None:
        vid = item.get("videoId") or item.get("id")
        title = item.get("title")
        if not vid or not title:
            return None
        artists = [a["name"] for a in (item.get("artists") or []) if a.get("name")]
        album = (item.get("album") or {}).get("name") if isinstance(item.get("album"), dict) else item.get("album")
        thumbs = item.get("thumbnails") or []
        return Track(
            source=self.name,
            source_id=vid,
            title=title,
            artists=artists,
            album=album,
            duration_sec=parse_duration(item.get("duration_seconds") or item.get("duration")),
            release_date=str(item.get("year")) if item.get("year") else None,
            play_count=parse_count(item.get("views")),
            thumbnail=thumbs[-1]["url"] if thumbs else None,
            url=f"https://music.youtube.com/watch?v={vid}",
            raw=item,
        )

    # -- collection --------------------------------------------------------------
    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        self.limiter.wait()
        try:
            results = self.client.search(query, filter="songs", limit=limit)
        except Exception as exc:
            LOG.warning("[ytm] search '%s' failed: %s", query, exc)
            return []

        tracks = [t for t in (self._to_track(r) for r in results) if t][:limit]
        return self._enrich_statistics(tracks)

    def chart_tracks(self, limit: int = 100) -> list[Track]:
        out: list[Track] = []
        seen: set[str] = set()

        def add_track(item: dict) -> None:
            if not isinstance(item, dict):
                return

            t = self._to_track(item)
            if t and t.uid not in seen:
                seen.add(t.uid)
                out.append(t)

        def extract_items(section: Any) -> list[dict]:
            """Normalize the various chart response shapes."""
            if not section:
                return []

            if isinstance(section, list):
                return [x for x in section if isinstance(x, dict)]

            if isinstance(section, dict):
                items = section.get("items")
                if isinstance(items, list):
                    return [x for x in items if isinstance(x, dict)]

            return []

        def extract_playlist_ids(section: Any) -> list[str]:
            """Find playlist IDs in a chart section."""
            ids: list[str] = []

            if isinstance(section, dict):
                # Older/current responses can expose one playlist directly.
                for key in ("playlist", "playlistId"):
                    value = section.get(key)
                    if isinstance(value, str) and value:
                        ids.append(value)

                # Or expose chart playlist entries under items.
                items = section.get("items")
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        for key in ("playlistId", "browseId"):
                            value = item.get(key)
                            if isinstance(value, str) and value:
                                # browseId can be VL<playlist-id>
                                if value.startswith("VL"):
                                    value = value[2:]
                                ids.append(value)

            elif isinstance(section, list):
                for item in section:
                    if not isinstance(item, dict):
                        continue

                    for key in ("playlistId", "browseId"):
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            if value.startswith("VL"):
                                value = value[2:]
                            ids.append(value)

            return list(dict.fromkeys(ids))

        for country in YTM_CHART_COUNTRIES:
            self.limiter.wait()
            try:
                charts = self.client.get_charts(country=country)

                if not isinstance(charts, dict):
                    LOG.warning(
                        "[ytm] charts %s returned unexpected type: %s",
                        country,
                        type(charts).__name__,
                    )
                    continue

                LOG.debug(
                    "[ytm] chart %s sections: %s",
                    country,
                    list(charts.keys()),
                )

                # ------------------------------------------------------------------
                # 1. Direct song charts
                # ------------------------------------------------------------------
                songs = charts.get("songs")

                for item in extract_items(songs):
                    add_track(item)

                # ------------------------------------------------------------------
                # 2. Video chart entries
                #
                # Some ytmusicapi versions return individual video/song items.
                # Others return chart playlist metadata.
                # ------------------------------------------------------------------
                videos = charts.get("videos")

                for item in extract_items(videos):
                    # If this is already a track, keep it.
                    if item.get("videoId") or item.get("id"):
                        add_track(item)

                # ------------------------------------------------------------------
                # 3. Resolve chart playlists.
                # ------------------------------------------------------------------
                playlist_ids = []

                for section_name in ("songs", "videos", "trending", "genres"):
                    playlist_ids.extend(
                        extract_playlist_ids(charts.get(section_name))
                    )

                # Remove duplicates while preserving order.
                playlist_ids = list(dict.fromkeys(playlist_ids))

                for playlist_id in playlist_ids:
                    if len(out) >= limit:
                        break

                    try:
                        self.limiter.wait()

                        playlist = self.client.get_playlist(
                            playlist_id,
                            limit=max(100, limit - len(out)),
                        )

                        for item in playlist.get("tracks", []):
                            if len(out) >= limit:
                                break

                            add_track(item)

                    except Exception as exc:
                        LOG.warning(
                            "[ytm] chart playlist %s failed: %s",
                            playlist_id,
                            exc,
                        )

                # ------------------------------------------------------------------
                # 4. Some versions expose useful chart items directly in
                #    "trending".
                # ------------------------------------------------------------------
                trending = charts.get("trending")

                for item in extract_items(trending):
                    if len(out) >= limit:
                        break
                    add_track(item)

                if out:
                    break

            except Exception as exc:
                LOG.warning(
                    "[ytm] charts %s failed: %s",
                    country,
                    exc,
                )

        LOG.info("[ytm] collected %d chart tracks", len(out))
        return self._enrich_statistics(out[:limit])

    def related_tracks(self, track: Track, limit: int = 20) -> list[Track]:
        """The watch-next queue is YouTube Music's own recommendation output —
        excellent supervision signal for an item-item model."""
        self.limiter.wait()
        try:
            watch = self.client.get_watch_playlist(videoId=track.source_id, limit=limit + 1)
        except Exception as exc:
            LOG.debug("[ytm] watch playlist failed for %s: %s", track.source_id, exc)
            return []
        out = []
        for item in watch.get("tracks", []):
            t = self._to_track(item)
            if t and t.source_id != track.source_id:
                out.append(t)
        return self._enrich_statistics(out[:limit])

    def playlists_for(self, track: Track, limit: int = 3) -> list[Playlist]:
        self.limiter.wait()
        try:
            results = self.client.search(
                f"{track.title} {track.artists[0] if track.artists else ''}",
                filter="playlists",
                limit=limit,
            )
        except Exception as exc:
            LOG.debug("[ytm] playlist search failed: %s", exc)
            return []
        playlists = []
        for r in results[:limit]:
            pid = r.get("browseId") or r.get("playlistId")
            if not pid:
                continue
            self.limiter.wait()
            try:
                detail = self.client.get_playlist(pid, limit=200)
            except Exception:
                continue
            uids = []
            for item in detail.get("tracks", []):
                t = self._to_track(item)
                if t:
                    uids.append(t.uid)
            playlists.append(
                Playlist(self.name, pid, detail.get("title", ""), detail.get("author", {}).get("name")
                         if isinstance(detail.get("author"), dict) else None, uids)
            )
        return playlists

# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    uid TEXT PRIMARY KEY, source TEXT, source_id TEXT, title TEXT, title_norm TEXT,
    artists TEXT, artists_norm TEXT, album TEXT, duration_sec INTEGER, genres TEXT,
    release_date TEXT, play_count INTEGER, like_count INTEGER, thumbnail TEXT,
    url TEXT, raw TEXT, fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_norm ON tracks(title_norm);
CREATE INDEX IF NOT EXISTS idx_tracks_source ON tracks(source);

CREATE TABLE IF NOT EXISTS playlists (
    uid TEXT PRIMARY KEY, source TEXT, source_id TEXT, title TEXT, owner TEXT,
    track_uids TEXT, fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    src_uid TEXT, dst_uid TEXT, kind TEXT, weight REAL,
    PRIMARY KEY (src_uid, dst_uid, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_uid);

CREATE TABLE IF NOT EXISTS matches (
    canonical_id TEXT, track_uid TEXT PRIMARY KEY, score REAL
);
CREATE INDEX IF NOT EXISTS idx_matches_canon ON matches(canonical_id);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def upsert_tracks(self, tracks: Iterable[Track]) -> int:
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for t in tracks:
            rows.append((
                t.uid, t.source, t.source_id, t.title, t.title_norm,
                json.dumps(t.artists, ensure_ascii=False), t.artists_norm, t.album,
                t.duration_sec, json.dumps(t.genres, ensure_ascii=False), t.release_date,
                t.play_count, t.like_count, t.thumbnail, t.url,
                json.dumps(t.raw, ensure_ascii=False)[:200_000], now,
            ))
        if not rows:
            return 0
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(uid) DO UPDATE SET "
                "  play_count=COALESCE(excluded.play_count, tracks.play_count),"
                "  like_count=COALESCE(excluded.like_count, tracks.like_count),"
                "  genres=CASE WHEN excluded.genres='[]' THEN tracks.genres ELSE excluded.genres END,"
                "  album=COALESCE(excluded.album, tracks.album),"
                "  duration_sec=COALESCE(excluded.duration_sec, tracks.duration_sec),"
                "  fetched_at=excluded.fetched_at",
                rows,
            )
        return len(rows)

    def upsert_playlists(self, playlists: Iterable[Playlist]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(p.uid, p.source, p.source_id, p.title, p.owner,
                 json.dumps(p.track_uids, ensure_ascii=False), now) for p in playlists]
        if not rows:
            return 0
        with self._lock, self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO playlists VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def upsert_edges(self, edges: Iterable[Edge]) -> int:
        rows = [(e.src_uid, e.dst_uid, e.kind, e.weight) for e in edges]
        if not rows:
            return 0
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT INTO edges VALUES (?,?,?,?) ON CONFLICT(src_uid, dst_uid, kind) "
                "DO UPDATE SET weight = edges.weight + excluded.weight", rows)
        return len(rows)

    def upsert_matches(self, pairs: Iterable[tuple[str, str, float]]) -> None:
        with self._lock, self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO matches VALUES (?,?,?)", list(pairs))

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def iter_tracks(self) -> Iterator[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        yield from self.conn.execute("SELECT * FROM tracks")

    def known_uids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT uid FROM tracks")}


# --------------------------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------------------------


class Crawler:
    def __init__(self, scraper: BaseScraper, store: Store, workers: int = 4):
        self.scraper = scraper
        self.store = store
        self.workers = workers

    def seed(self, *, use_charts: bool, queries: Sequence[str], per_source: int) -> list[Track]:
        seeds: list[Track] = []
        if use_charts:
            try:
                got = self.scraper.chart_tracks(limit=per_source)
                LOG.info("[youtube] %d chart tracks", len(got))
                seeds.extend(got)
            except NotImplementedError:
                pass
            except Exception as exc:
                LOG.warning("[youtube] charts failed: %s", exc)
        for q in queries:
            try:
                got = self.scraper.search_tracks(q, limit=min(per_source, 50))
                LOG.info("[youtube] '%s' -> %d tracks", q, len(got))
                seeds.extend(got)
            except Exception as exc:
                LOG.warning("[youtube] search '%s' failed: %s", q, exc)
        self.store.upsert_tracks(seeds)
        return seeds

    def expand(self, seeds: Sequence[Track], *, depth: int, max_tracks: int,
               fanout: int = 15, collect_playlists: bool = False) -> None:
        """BFS over each source's own related-items graph."""
        visited = self.store.known_uids()
        queue: deque[tuple[Track, int]] = deque((t, 0) for t in seeds)
        processed = 0

        while queue and self.store.count("tracks") < max_tracks:
            batch: list[tuple[Track, int]] = []
            while queue and len(batch) < self.workers * 2:
                batch.append(queue.popleft())

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {}
                for track, level in batch:
                    if level >= depth:
                        continue
                    future = pool.submit(self._expand_one, track, fanout, collect_playlists)
                    futures[future] = (track, level)

                for fut in as_completed(futures):
                    track, level = futures[fut]
                    try:
                        neighbours, playlists = fut.result()
                    except Exception as exc:
                        LOG.debug("expand failed for %s: %s", track.uid, exc)
                        continue

                    self.store.upsert_tracks(neighbours)
                    self.store.upsert_edges(
                        Edge(track.uid, n.uid, "related", weight=1.0 / (rank + 1))
                        for rank, n in enumerate(neighbours)
                    )
                    if playlists:
                        self.store.upsert_playlists(playlists)
                        self.store.upsert_edges(self._cooccurrence(playlists))

                    for n in neighbours:
                        if n.uid not in visited:
                            visited.add(n.uid)
                            queue.append((n, level + 1))

                    processed += 1
                    if processed % 25 == 0:
                        LOG.info("expanded %d nodes | tracks=%d edges=%d",
                                 processed, self.store.count("tracks"), self.store.count("edges"))

    def _expand_one(self, track: Track, fanout: int,
                    collect_playlists: bool) -> tuple[list[Track], list[Playlist]]:
        neighbours = self.scraper.related_tracks(track, limit=fanout)
        playlists = self.scraper.playlists_for(track) if collect_playlists else []
        return neighbours, playlists

    @staticmethod
    def _cooccurrence(playlists: Sequence[Playlist], cap: int = 60) -> list[Edge]:
        """Co-occurrence weighted by 1/log(playlist size) so 500-track dumps don't
        drown out tight, curated lists."""
        import math
        edges: list[Edge] = []
        for p in playlists:
            uids = p.track_uids[:cap]
            if len(uids) < 2:
                continue
            w = 1.0 / math.log(len(uids) + 1)
            for i, a in enumerate(uids):
                for b in uids[i + 1:]:
                    edges.append(Edge(a, b, "playlist_cooccur", w))
                    edges.append(Edge(b, a, "playlist_cooccur", w))
        return edges

# --------------------------------------------------------------------------------------
# Export for the recommender
# --------------------------------------------------------------------------------------


def export(store: Store, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    canon = {uid: cid for cid, uid, _ in
             store.conn.execute("SELECT canonical_id, track_uid, score FROM matches")}

    # items.jsonl — content features
    items_path = out_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as fh:
        for r in store.iter_tracks():
            artists = json.loads(r["artists"] or "[]")
            fh.write(json.dumps({
                "item_id": canon.get(r["uid"], r["uid"]),
                "source_uid": r["uid"],
                "source": r["source"],
                "title": r["title"],
                "title_norm": r["title_norm"],
                "artists": artists,
                "primary_artist": artists[0] if artists else None,
                "album": r["album"],
                "genres": json.loads(r["genres"] or "[]"),
                "duration_sec": r["duration_sec"],
                "release_date": r["release_date"],
                "play_count": r["play_count"],
                "like_count": r["like_count"],
                "popularity_log": round(__import__("math").log1p(r["play_count"] or 0), 4),
                "url": r["url"],
                "thumbnail": r["thumbnail"],
                # ready-to-embed text field for a sentence-transformer / TF-IDF pass
                "text": " | ".join(filter(None, [
                    r["title"], ", ".join(artists), r["album"],
                    " ".join(json.loads(r["genres"] or "[]")),
                ])),
            }, ensure_ascii=False) + "\n")

    # edges.csv — item-item graph, canonicalised and deduped
    edges_path = out_dir / "item_item_edges.csv"
    agg: dict[tuple[str, str, str], float] = {}
    for src, dst, kind, w in store.conn.execute("SELECT src_uid, dst_uid, kind, weight FROM edges"):
        a, b = canon.get(src, src), canon.get(dst, dst)
        if a == b:
            continue
        agg[(a, b, kind)] = agg.get((a, b, kind), 0.0) + w
    with edges_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_item", "target_item", "kind", "weight"])
        for (a, b, kind), w in agg.items():
            writer.writerow([a, b, kind, round(w, 6)])

    # playlists.jsonl — sequence data for session-based / word2vec-style models
    pl_path = out_dir / "playlists.jsonl"
    with pl_path.open("w", encoding="utf-8") as fh:
        for uid, source, title, track_uids in store.conn.execute(
                "SELECT uid, source, title, track_uids FROM playlists"):
            items = [canon.get(u, u) for u in json.loads(track_uids or "[]")]
            if len(items) < 2:
                continue
            fh.write(json.dumps({"playlist_id": uid, "source": source, "title": title,
                                 "items": items}, ensure_ascii=False) + "\n")

    LOG.info("exported -> %s (%d items, %d edges)", out_dir, store.count("tracks"), len(agg))
    print(f"\n  items      : {items_path}\n  edges      : {edges_path}\n  playlists  : {pl_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "nhạc trẻ hay nhất", "v-pop 2024", "ballad việt buồn", "rap việt", "indie việt",
    "nhạc chill", "lofi việt", "bolero", "nhạc remix hot", "acoustic cover việt",
]

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="music.db")
    p.add_argument("--cache-dir", default=".http_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--ignore-robots", action="store_true", help="skip robots.txt checks")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="collect tracks and relationship edges")
    c.add_argument("--queries", nargs="*", default=None)
    c.add_argument("--seed-charts", action="store_true")
    c.add_argument("--per-source", type=int, default=100)
    c.add_argument("--depth", type=int, default=1)
    c.add_argument("--fanout", type=int, default=15)
    c.add_argument("--max-tracks", type=int, default=5000)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--playlists", action="store_true", help="also harvest playlist co-occurrence")
    c.add_argument("--ytm-auth", default=os.getenv("YTM_AUTH", "browser.json"))

    e = sub.add_parser("export", help="write recommender training files")
    e.add_argument("--out", default="./dataset")

    sub.add_parser("stats", help="show what's in the database")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    store = Store(args.db)

    if args.cmd == "crawl":
        cache = HttpCache(Path(args.cache_dir), enabled=not args.no_cache)
        robots = RobotsGate(enabled=not args.ignore_robots)
        try:
            scraper = YouTubeMusicScraper(cache, robots, args.ytm_auth)
        except Exception as exc:
            LOG.error("could not initialize YouTube Music: %s", exc)
            return 1

        queries = args.queries if args.queries is not None else (
            [] if args.seed_charts else DEFAULT_QUERIES)
        crawler = Crawler(scraper, store, workers=args.workers)
        seeds = crawler.seed(use_charts=args.seed_charts, queries=queries,
                             per_source=args.per_source)
        LOG.info("seeded %d tracks (db now %d)", len(seeds), store.count("tracks"))
        if args.depth > 0:
            crawler.expand(seeds, depth=args.depth, max_tracks=args.max_tracks,
                           fanout=args.fanout, collect_playlists=args.playlists)

    elif args.cmd == "export":
        export(store, Path(args.out))

    elif args.cmd == "stats":
        for table in ("tracks", "playlists", "edges", "matches"):
            print(f"{table:<12} {store.count(table):>8}")
        print("\nby source:")
        for source, n in store.conn.execute(
                "SELECT source, COUNT(*) FROM tracks GROUP BY source ORDER BY 2 DESC"):
            print(f"  {source:<16} {n:>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
