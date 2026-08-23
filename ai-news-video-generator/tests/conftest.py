"""
Shared fixtures for the AI NEWS Generator test suite.
"""

import pytest


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
    <header><nav>Menu</nav></header>
    <script>var x = 1;</script>
    <style>.body { color: red; }</style>
    <article>
        <p>NASA confirmed that its Artemis IV mission has successfully
           entered lunar orbit today.</p>
        <p>The crew of four astronauts reported all systems nominal.</p>
        <p>International partners ESA, JAXA, and ISRO contributed key
           instruments aboard the lander.</p>
    </article>
    <footer>Copyright 2026</footer>
</body>
</html>
"""

SAMPLE_GEMINI_RESPONSE = """
[
  {
    "scene_number": 1,
    "timestamp": "00:00 - 00:15",
    "narration": "NASA's Artemis IV has entered lunar orbit.",
    "visual_prompt": "Cinematic wide shot of spacecraft orbiting Moon."
  },
  {
    "scene_number": 2,
    "timestamp": "00:15 - 00:30",
    "narration": "Four astronauts prepare for surface expedition.",
    "visual_prompt": "Astronauts inside cockpit checking instruments."
  },
  {
    "scene_number": 3,
    "timestamp": "00:30 - 00:60",
    "narration": "The mission aims to extract water-ice samples.",
    "visual_prompt": "Robotic drill on dark lunar crater surface."
  }
]
"""

SAMPLE_GEMINI_RESPONSE_FENCED = """
Here is the storyboard:

```json
[
  {
    "scene_number": 1,
    "timestamp": "00:00 - 00:30",
    "narration": "Breaking news from lunar orbit.",
    "visual_prompt": "Spacecraft approaching the Moon."
  }
]
```
"""


@pytest.fixture
def sample_html():
    return SAMPLE_HTML


@pytest.fixture
def sample_gemini_response():
    return SAMPLE_GEMINI_RESPONSE


@pytest.fixture
def sample_gemini_response_fenced():
    return SAMPLE_GEMINI_RESPONSE_FENCED


@pytest.fixture
def sample_article_text():
    return (
        "NASA confirmed that its Artemis IV mission has successfully entered "
        "lunar orbit. The crew of four astronauts reported all systems nominal. "
        "International partners ESA, JAXA, and ISRO contributed key instruments."
    )
