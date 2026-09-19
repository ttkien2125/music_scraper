#!/usr/bin/env python3
"""
music_scraper.py — Multi-source music metadata collector for a recommendation system.

Sources
-------
  * YouTube Music  (via the `ytmusicapi` unofficial client)
  * ZingMP3        (via its public web JSON API, HMAC-signed)
  * NhacCuaTui     (via HTML pages + embedded JSON)

What it collects
----------------
  1. Track metadata      -> content-based features (title, artists, genre, duration, year)
  2. Playlist membership -> co-occurrence signal (the strongest free CF signal you can get)
  3. "Related track" edges -> item-item graph, ready for random-walk / ALS style models
  4. Cross-source track matching -> merges the same song across the three catalogs

Install
-------
    pip install requests beautifulsoup4 lxml ytmusicapi rapidfuzz tenacity

Usage
-----
    # Seed from charts on every source, then expand 2 hops through related tracks
    python music_scraper.py crawl --sources all --seed-charts --depth 2 --max-tracks 5000

    # Seed from your own search terms
    python music_scraper.py crawl --sources zingmp3,nhaccuatui \
        --queries "Sơn Tùng M-TP" "Hoàng Thùy Linh" "indie việt" --max-tracks 800

    # Link the same song across sources
    python music_scraper.py match --threshold 88

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
import hmac
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
import urllib.robotparser as robotparser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from ytmusicapi import YTMusic
except ImportError:  # pragma: no cover
    YTMusic = None

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


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


class ZingConfig:
    """ZingMP3's web client signs every request. These values come from its public JS
    bundle and DO rotate — if you start getting `{"err": -201}` re-read them from
    https://zingmp3.vn/ main chunk and update, or set them via env vars."""

    BASE = "https://zingmp3.vn"
    API_KEY = os.getenv("ZING_API_KEY", "88265e23d4284f25963e6eedac8fbfa3")
    SECRET_KEY = os.getenv("ZING_SECRET_KEY", "2aa2d1c561e809b267f3638c4a307aab")
    VERSION = os.getenv("ZING_VERSION", "1.13.13")

    PATH_SONG_INFO = "/api/v2/song/get/info"
    PATH_SEARCH = "/api/v2/search"
    PATH_CHART_HOME = "/api/v2/page/get/chart-home"
    PATH_PLAYLIST = "/api/v2/page/get/playlist"
    PATH_ARTIST = "/api/v2/page/get/artist"


class NctConfig:
    BASE = "https://www.nhaccuatui.com"
    CHART_URLS = [
        "https://www.nhaccuatui.com/bai-hat/top-100-nhac-tre-hay-nhat.html",
        "https://www.nhaccuatui.com/bai-hat/top-100-nhac-viet.html",
        "https://www.nhaccuatui.com/top-hits.html",
    ]
    SEARCH_URL = "https://www.nhaccuatui.com/tim-kiem/bai-hat?q={q}"
    # Song pages embed a JS object; these regexes pull it out without a JS engine.
    RE_PLAYER_JSON = re.compile(r"player\.peConfig\s*=\s*(\{.*?\});", re.S)
    RE_XML_KEY = re.compile(r'"?key"?\s*:\s*"([a-f0-9]{16,})"')
    RE_SONG_HREF = re.compile(r"/bai-hat/[^\"']+\.html")


YTM_CHART_COUNTRIES = ["VN", "US", "ZZ"]  # ZZ == global


# --------------------------------------------------------------------------------------
# Unified data model
# --------------------------------------------------------------------------------------


@dataclass
class Track:
    source: str                      # youtube_music | zingmp3 | nhaccuatui
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
    kind: str          # related | playlist_cooccur | same_artist | same_album
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
        self.client = YTMusic(auth_file) if auth_file and Path(auth_file).exists() else YTMusic()

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
        return [t for t in (self._to_track(r) for r in results) if t][:limit]

    def chart_tracks(self, limit: int = 100) -> list[Track]:
        out: list[Track] = []
        for country in YTM_CHART_COUNTRIES:
            self.limiter.wait()
            try:
                charts = self.client.get_charts(country=country)
            except Exception as exc:
                LOG.warning("[ytm] charts %s failed: %s", country, exc)
                continue
            for bucket in ("songs", "videos", "trending"):
                section = charts.get(bucket) or {}
                for item in (section.get("items") if isinstance(section, dict) else section) or []:
                    t = self._to_track(item)
                    if t:
                        out.append(t)
        return out[:limit]

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
        return out[:limit]

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
# ZingMP3
# --------------------------------------------------------------------------------------


class ZingMp3Scraper(BaseScraper):
    name = "zingmp3"

    def __init__(self, cache: HttpCache, robots: RobotsGate):
        super().__init__(cache, robots, min_interval=1.2)
        self.session.headers.update({"Referer": ZingConfig.BASE + "/", "Origin": ZingConfig.BASE})
        self._bootstrap_cookies()

    def _bootstrap_cookies(self) -> None:
        """Zing sets a zmp3_rqid cookie on the landing page; without it some
        endpoints return -201."""
        try:
            self.limiter.wait()
            self.session.get(ZingConfig.BASE + "/", timeout=20)
        except requests.RequestException as exc:
            LOG.debug("[zing] cookie bootstrap failed: %s", exc)

    # -- request signing ----------------------------------------------------------
    @staticmethod
    def _sha256(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    def _sign(self, path: str, params: dict[str, Any]) -> str:
        payload = "".join(f"{k}={params[k]}" for k in sorted(params))
        return hmac.new(
            ZingConfig.SECRET_KEY.encode(),
            (path + self._sha256(payload)).encode(),
            hashlib.sha512,
        ).hexdigest()

    def _api(self, path: str, **params: Any) -> dict:
        signed = {"ctime": str(int(time.time())), "version": ZingConfig.VERSION, **params}
        query = {**signed, "sig": self._sign(path, signed), "apiKey": ZingConfig.API_KEY}
        data = self.fetch(ZingConfig.BASE + path, params=query, as_json=True,
                          cache_key=f"zing:{path}:{json.dumps(params, sort_keys=True)}")
        if data.get("err") not in (0, None):
            raise RuntimeError(f"ZingMP3 error {data.get('err')}: {data.get('msg')} ({path})")
        return data.get("data") or {}

    # -- mapping -------------------------------------------------------------------
    def _to_track(self, item: dict) -> Track | None:
        sid = item.get("encodeId") or item.get("id")
        title = item.get("title")
        if not sid or not title:
            return None
        artists = [a["name"] for a in (item.get("artists") or []) if a.get("name")]
        if not artists and item.get("artistsNames"):
            artists = [a.strip() for a in str(item["artistsNames"]).split(",") if a.strip()]
        album = (item.get("album") or {}).get("title") if isinstance(item.get("album"), dict) else None
        genres = [g["name"] for g in (item.get("genres") or []) if g.get("name")]
        release = item.get("releaseDate")
        if isinstance(release, (int, float)) and release > 0:
            release = datetime.fromtimestamp(release, tz=timezone.utc).strftime("%Y-%m-%d")
        return Track(
            source=self.name,
            source_id=sid,
            title=title,
            artists=artists,
            album=album,
            duration_sec=parse_duration(item.get("duration")),
            genres=genres,
            release_date=str(release) if release else None,
            play_count=parse_count(item.get("listen") or (item.get("streamingStatus") and None)),
            like_count=parse_count(item.get("like")),
            thumbnail=item.get("thumbnailM") or item.get("thumbnail"),
            url=urljoin(ZingConfig.BASE, item.get("link", "")) if item.get("link") else None,
            raw=item,
        )

    def _walk_songs(self, node: Any) -> Iterator[dict]:
        """Zing nests song lists at unpredictable depths; harvest them all."""
        if isinstance(node, dict):
            if node.get("encodeId") and node.get("title") and "duration" in node:
                yield node
            for v in node.values():
                yield from self._walk_songs(v)
        elif isinstance(node, list):
            for v in node:
                yield from self._walk_songs(v)

    # -- collection -----------------------------------------------------------------
    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        try:
            data = self._api(ZingConfig.PATH_SEARCH, q=query, type="song", page=1, count=limit)
        except Exception as exc:
            LOG.warning("[zing] search '%s' failed: %s", query, exc)
            return []
        items = data.get("items") or data.get("songs") or list(self._walk_songs(data))
        return [t for t in (self._to_track(i) for i in items) if t][:limit]

    def chart_tracks(self, limit: int = 100) -> list[Track]:
        try:
            data = self._api(ZingConfig.PATH_CHART_HOME)
        except Exception as exc:
            LOG.warning("[zing] chart-home failed: %s", exc)
            return []
        seen, out = set(), []
        for item in self._walk_songs(data):
            t = self._to_track(item)
            if t and t.uid not in seen:
                seen.add(t.uid)
                out.append(t)
        return out[:limit]

    def related_tracks(self, track: Track, limit: int = 20) -> list[Track]:
        """The song-info payload carries Zing's own 'recommends' / artist sections."""
        try:
            data = self._api(ZingConfig.PATH_SONG_INFO, id=track.source_id)
        except Exception as exc:
            LOG.debug("[zing] song info failed for %s: %s", track.source_id, exc)
            return []
        out, seen = [], {track.source_id}
        for item in self._walk_songs(data):
            t = self._to_track(item)
            if t and t.source_id not in seen:
                seen.add(t.source_id)
                out.append(t)
        return out[:limit]

    def playlist(self, playlist_id: str) -> Playlist | None:
        try:
            data = self._api(ZingConfig.PATH_PLAYLIST, id=playlist_id)
        except Exception:
            return None
        uids = [f"{self.name}:{s['encodeId']}" for s in self._walk_songs(data) if s.get("encodeId")]
        return Playlist(self.name, playlist_id, data.get("title", ""), (data.get("artist") or {}).get("name"), uids)


# --------------------------------------------------------------------------------------
# NhacCuaTui
# --------------------------------------------------------------------------------------


class NhacCuaTuiScraper(BaseScraper):
    name = "nhaccuatui"

    def __init__(self, cache: HttpCache, robots: RobotsGate):
        super().__init__(cache, robots, min_interval=1.5)
        if BeautifulSoup is None:
            raise RuntimeError("pip install beautifulsoup4 lxml to use the NhacCuaTui source")

    @staticmethod
    def _soup(html: str) -> "BeautifulSoup":
        try:
            return BeautifulSoup(html, "lxml")
        except Exception:
            return BeautifulSoup(html, "html.parser")

    @staticmethod
    def _song_id_from_url(url: str) -> str | None:
        m = re.search(r"/bai-hat/[^/]*?\.([A-Za-z0-9_-]+)\.html", url)
        if m:
            return m.group(1)
        slug = urlparse(url).path.rstrip("/").split("/")[-1].replace(".html", "")
        return slug or None

    def _parse_song_page(self, url: str) -> Track | None:
        try:
            html = self.fetch(url)
        except Exception as exc:
            LOG.debug("[nct] fetch %s failed: %s", url, exc)
            return None
        soup = self._soup(html)

        title, artists, album, duration, genres, release, thumb, plays = (
            None, [], None, None, [], None, None, None,
        )

        # 1) Schema.org JSON-LD is the most stable path when present.
        for tag in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                blob = json.loads(tag.string or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            for node in (blob if isinstance(blob, list) else [blob]):
                if not isinstance(node, dict):
                    continue
                if node.get("@type") in ("MusicRecording", "VideoObject", "AudioObject"):
                    title = title or node.get("name")
                    by = node.get("byArtist")
                    if isinstance(by, dict):
                        artists = artists or [by.get("name")]
                    elif isinstance(by, list):
                        artists = artists or [b.get("name") for b in by if isinstance(b, dict)]
                    album = album or (node.get("inAlbum") or {}).get("name") if isinstance(
                        node.get("inAlbum"), dict) else album
                    thumb = thumb or node.get("thumbnailUrl") or node.get("image")
                    release = release or node.get("uploadDate") or node.get("datePublished")
                    if node.get("duration"):
                        m = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", str(node["duration"]))
                        if m:
                            duration = int(m.group(1) or 0) * 60 + int(m.group(2) or 0)

        # 2) Fall back to meta tags + visible DOM.
        if not title:
            og = soup.find("meta", property="og:title")
            title = og["content"].strip() if og and og.get("content") else (
                soup.find("h1").get_text(strip=True) if soup.find("h1") else None)
        if not thumb:
            og_img = soup.find("meta", property="og:image")
            thumb = og_img["content"] if og_img and og_img.get("content") else None
        if not artists:
            for sel in (".name_singer a", ".singer_song a", "h2.name_singer a", 'a[href*="/nghe-si/"]'):
                found = [a.get_text(strip=True) for a in soup.select(sel) if a.get_text(strip=True)]
                if found:
                    artists = found
                    break
        for sel, bucket in ((".name_cate a", "genres"), ('a[href*="/the-loai/"]', "genres")):
            vals = [a.get_text(strip=True) for a in soup.select(sel)]
            if vals:
                genres = vals
                break
        listen = soup.find(string=re.compile(r"lượt nghe", re.I))
        if listen:
            plays = parse_count(str(listen))

        # 3) Player config JSON carries duration / stream key when the DOM doesn't.
        m = NctConfig.RE_PLAYER_JSON.search(html)
        if m:
            try:
                cfg = json.loads(m.group(1))
                duration = duration or parse_duration(cfg.get("duration"))
            except json.JSONDecodeError:
                pass

        if not title:
            return None
        sid = self._song_id_from_url(url)
        if not sid:
            return None
        return Track(
            source=self.name,
            source_id=sid,
            title=title,
            artists=[a for a in artists if a],
            album=album,
            duration_sec=duration,
            genres=genres,
            release_date=str(release)[:10] if release else None,
            play_count=plays,
            thumbnail=thumb,
            url=url,
            raw={"scraped_from": url},
        )

    def _song_links(self, html: str, limit: int) -> list[str]:
        soup = self._soup(html)
        links, seen = [], set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not NctConfig.RE_SONG_HREF.search(href):
                continue
            full = urljoin(NctConfig.BASE, href)
            if full in seen:
                continue
            seen.add(full)
            links.append(full)
            if len(links) >= limit:
                break
        return links

    def chart_tracks(self, limit: int = 100) -> list[Track]:
        links: list[str] = []
        for chart_url in NctConfig.CHART_URLS:
            try:
                html = self.fetch(chart_url)
            except Exception as exc:
                LOG.warning("[nct] chart %s failed: %s", chart_url, exc)
                continue
            links.extend(self._song_links(html, limit))
            if len(links) >= limit:
                break
        return [t for t in (self._parse_song_page(u) for u in links[:limit]) if t]

    def search_tracks(self, query: str, limit: int = 20) -> list[Track]:
        url = NctConfig.SEARCH_URL.format(q=requests.utils.quote(query))
        try:
            html = self.fetch(url)
        except Exception as exc:
            LOG.warning("[nct] search '%s' failed: %s", query, exc)
            return []
        return [t for t in (self._parse_song_page(u) for u in self._song_links(html, limit)) if t]

    def related_tracks(self, track: Track, limit: int = 20) -> list[Track]:
        """Song pages carry a 'Có thể bạn muốn nghe' block — that's an editorial
        related-items list, ideal as graph edges."""
        if not track.url:
            return []
        try:
            html = self.fetch(track.url)
        except Exception:
            return []
        links = [u for u in self._song_links(html, limit * 2) if u != track.url]
        return [t for t in (self._parse_song_page(u) for u in links[:limit]) if t]


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
    def __init__(self, scrapers: dict[str, BaseScraper], store: Store, workers: int = 4):
        self.scrapers = scrapers
        self.store = store
        self.workers = workers

    def seed(self, *, use_charts: bool, queries: Sequence[str], per_source: int) -> list[Track]:
        seeds: list[Track] = []
        for name, sc in self.scrapers.items():
            if use_charts:
                try:
                    got = sc.chart_tracks(limit=per_source)
                    LOG.info("[%s] %d chart tracks", name, len(got))
                    seeds.extend(got)
                except NotImplementedError:
                    pass
                except Exception as exc:
                    LOG.warning("[%s] charts failed: %s", name, exc)
            for q in queries:
                try:
                    got = sc.search_tracks(q, limit=min(per_source, 50))
                    LOG.info("[%s] '%s' -> %d tracks", name, q, len(got))
                    seeds.extend(got)
                except Exception as exc:
                    LOG.warning("[%s] search '%s' failed: %s", name, q, exc)
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
                    scraper = self.scrapers.get(track.source)
                    if scraper is None:
                        continue
                    futures[pool.submit(self._expand_one, scraper, track, fanout,
                                        collect_playlists)] = (track, level)

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

    @staticmethod
    def _expand_one(scraper: BaseScraper, track: Track, fanout: int,
                    collect_playlists: bool) -> tuple[list[Track], list[Playlist]]:
        neighbours = scraper.related_tracks(track, limit=fanout)
        playlists = scraper.playlists_for(track) if collect_playlists else []
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
# Cross-source matching
# --------------------------------------------------------------------------------------


def _ratio(a: str, b: str) -> float:
    if fuzz is not None:
        return float(fuzz.token_set_ratio(a, b))
    # Fallback: token Jaccard * 100
    sa, sb = set(a.split()), set(b.split())
    return 100.0 * len(sa & sb) / max(1, len(sa | sb))


def match_across_sources(store: Store, threshold: float = 88.0,
                         duration_tolerance: int = 6) -> int:
    """Blocks candidates by a coarse title key, then scores title+artist+duration.
    Produces a `canonical_id` so the recommender treats one song as one item."""
    rows = list(store.iter_tracks())
    blocks: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        key = " ".join(sorted((r["title_norm"] or "").split())[:3])
        blocks.setdefault(key, []).append(r)

    parent: dict[str, str] = {r["uid"]: r["uid"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    scores: dict[str, float] = {}
    for group in blocks.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a["source"] == b["source"]:
                    continue
                title_score = _ratio(a["title_norm"] or "", b["title_norm"] or "")
                if title_score < threshold - 5:
                    continue
                artist_score = _ratio(a["artists_norm"] or "", b["artists_norm"] or "")
                score = 0.65 * title_score + 0.35 * artist_score
                da, db = a["duration_sec"], b["duration_sec"]
                if da and db:
                    score += 5 if abs(da - db) <= duration_tolerance else -15
                if score >= threshold:
                    union(a["uid"], b["uid"])
                    scores[a["uid"]] = max(scores.get(a["uid"], 0), score)
                    scores[b["uid"]] = max(scores.get(b["uid"], 0), score)

    store.upsert_matches((find(r["uid"]), r["uid"], scores.get(r["uid"], 100.0)) for r in rows)
    clusters = len({find(r["uid"]) for r in rows})
    LOG.info("matched %d rows into %d canonical items", len(rows), clusters)
    return clusters


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


def build_scrapers(names: Sequence[str], cache: HttpCache, robots: RobotsGate,
                   ytm_auth: str | None) -> dict[str, BaseScraper]:
    factories = {
        "youtube_music": lambda: YouTubeMusicScraper(cache, robots, ytm_auth),
        "zingmp3": lambda: ZingMp3Scraper(cache, robots),
        "nhaccuatui": lambda: NhacCuaTuiScraper(cache, robots),
    }
    out: dict[str, BaseScraper] = {}
    for n in names:
        try:
            out[n] = factories[n]()
            LOG.info("source ready: %s", n)
        except KeyError:
            LOG.error("unknown source: %s", n)
        except Exception as exc:
            LOG.error("could not init %s: %s", n, exc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="music.db")
    p.add_argument("--cache-dir", default=".http_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--ignore-robots", action="store_true", help="skip robots.txt checks")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="collect tracks and relationship edges")
    c.add_argument("--sources", default="all",
                   help="comma list of youtube_music,zingmp3,nhaccuatui or 'all'")
    c.add_argument("--queries", nargs="*", default=None)
    c.add_argument("--seed-charts", action="store_true")
    c.add_argument("--per-source", type=int, default=100)
    c.add_argument("--depth", type=int, default=1)
    c.add_argument("--fanout", type=int, default=15)
    c.add_argument("--max-tracks", type=int, default=5000)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--playlists", action="store_true", help="also harvest playlist co-occurrence")
    c.add_argument("--ytm-auth", default=os.getenv("YTM_AUTH", "browser.json"))

    m = sub.add_parser("match", help="link the same song across sources")
    m.add_argument("--threshold", type=float, default=88.0)

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
        names = (["youtube_music", "zingmp3", "nhaccuatui"] if args.sources == "all"
                 else [s.strip() for s in args.sources.split(",") if s.strip()])
        cache = HttpCache(Path(args.cache_dir), enabled=not args.no_cache)
        robots = RobotsGate(enabled=not args.ignore_robots)
        scrapers = build_scrapers(names, cache, robots, args.ytm_auth)
        if not scrapers:
            LOG.error("no usable sources; aborting")
            return 1

        queries = args.queries if args.queries is not None else (
            [] if args.seed_charts else DEFAULT_QUERIES)
        crawler = Crawler(scrapers, store, workers=args.workers)
        seeds = crawler.seed(use_charts=args.seed_charts, queries=queries,
                             per_source=args.per_source)
        LOG.info("seeded %d tracks (db now %d)", len(seeds), store.count("tracks"))
        if args.depth > 0:
            crawler.expand(seeds, depth=args.depth, max_tracks=args.max_tracks,
                           fanout=args.fanout, collect_playlists=args.playlists)

    elif args.cmd == "match":
        match_across_sources(store, threshold=args.threshold)

    elif args.cmd == "export":
        if store.count("matches") == 0:
            LOG.info("no match table yet — running matcher first")
            match_across_sources(store)
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
