"""Reddit data ingestion via PRAW with rate limiting and pagination."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Generator, List, Optional

import praw
from praw.models import Comment, Submission
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models import ContentType, RedditItem


class RedditScraper:
  def __init__(self) -> None:
    settings = get_settings()
    self.settings = settings
    self.reddit = praw.Reddit(
      client_id=settings.reddit_client_id,
      client_secret=settings.reddit_client_secret,
      user_agent=settings.reddit_user_agent,
    )
    self._request_delay = 1.0

  def scrape_all_windows(self) -> List[RedditItem]:
    all_items: List[RedditItem] = []
    for label, start, end in self.settings.time_windows:
      print(f"  Scraping window {label} ({start} -> {end})...")
      items = list(self.scrape_window(label, start, end))
      print(f"    -> {len(items)} items")
      all_items.extend(items)
    return all_items

  def scrape_window(
    self, label: str, start_iso: str, end_iso: str
  ) -> Generator[RedditItem, None, None]:
    start_ts = datetime.fromisoformat(start_iso).timestamp()
    end_ts = datetime.fromisoformat(f"{end_iso}T23:59:59").timestamp()

    for subreddit_name in self.settings.subreddit_list:
      yield from self._scrape_subreddit(subreddit_name, label, start_ts, end_ts)
      time.sleep(self._request_delay)

  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
  def _scrape_subreddit(
    self, subreddit_name: str, label: str, start_ts: float, end_ts: float
  ) -> Generator[RedditItem, None, None]:
    subreddit = self.reddit.subreddit(subreddit_name)
    count = 0

    for submission in subreddit.search(
      self.settings.reddit_search_query,
      sort="new",
      time_filter="all",
      limit=self.settings.reddit_post_limit * 3,
    ):
      if submission.created_utc < start_ts or submission.created_utc > end_ts:
        continue

      yield self._submission_to_item(submission, label)
      count += 1
      if count >= self.settings.reddit_post_limit:
        break

      submission.comments.replace_more(limit=0)
      comment_count = 0
      for comment in submission.comments.list():
        if comment.created_utc < start_ts or comment.created_utc > end_ts:
          continue
        yield self._comment_to_item(comment, submission, label)
        comment_count += 1
        if comment_count >= self.settings.reddit_comment_limit:
          break

      time.sleep(0.5)

  @staticmethod
  def _submission_to_item(sub: Submission, label: str) -> RedditItem:
    return RedditItem(
      id=f"post_{sub.id}",
      content_type=ContentType.POST,
      title=sub.title,
      body=sub.selftext or "",
      author=str(sub.author) if sub.author else "[deleted]",
      subreddit=str(sub.subreddit),
      created_utc=sub.created_utc,
      score=sub.score,
      url=sub.permalink,
      post_id=sub.id,
      window_label=label,
    )

  @staticmethod
  def _comment_to_item(
    comment: Comment, submission: Submission, label: str
  ) -> RedditItem:
    parent_id = comment.parent_id
    return RedditItem(
      id=f"comment_{comment.id}",
      content_type=ContentType.COMMENT,
      title="",
      body=comment.body,
      author=str(comment.author) if comment.author else "[deleted]",
      subreddit=str(submission.subreddit),
      created_utc=comment.created_utc,
      score=comment.score,
      url=comment.permalink,
      parent_id=parent_id,
      post_id=submission.id,
      window_label=label,
    )
