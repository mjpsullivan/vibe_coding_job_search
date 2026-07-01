import asyncio
import io
import json
import os

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract concatenated text from every page of a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
            }
        )
    return jobs


async def fetch_all_jobs() -> list[dict]:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
            return_exceptions=True,
        )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)

    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    jobs = await fetch_all_jobs()
    return {"count": len(jobs), "jobs": jobs}


async def score_jobs_with_gemini(cv: str, role: str, jobs: list[dict]) -> list[dict]:
    """Ask Gemini to score each job's fit against a CV and desired role."""
    api_key = os.environ["GEMINI_API_KEY"]

    job_list = [
        {"index": i, "title": job["title"], "company": job["company"], "location": job["location"]}
        for i, job in enumerate(jobs)
    ]

    prompt = (
        "You are a job-matching assistant. Given a candidate's CV, the role they want, "
        "and a list of open job postings, score how well each job fits the candidate.\n\n"
        f"Desired role: {role}\n\n"
        f"Candidate CV:\n{cv}\n\n"
        f"Jobs:\n{json.dumps(job_list)}\n\n"
        "Score every job from 0 to 100 (100 = perfect fit) and give a short one-sentence "
        "reason for each score."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "index": {"type": "INTEGER"},
                        "score": {"type": "INTEGER"},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["index", "score", "reason"],
                },
            },
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(GEMINI_URL, params={"key": api_key}, json=payload)
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini request failed: {response.status_code} {response.text}",
            )
        data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    scores = json.loads(text)

    ranked = []
    for item in scores:
        job = jobs[item["index"]]
        ranked.append(
            {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "apply_url": job["apply_url"],
                "score": item["score"],
                "reason": item["reason"],
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


@app.post("/rank")
async def rank_jobs(cv: UploadFile = File(...), role: str = Form(...)) -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    cv_bytes = await cv.read()
    cv_text = extract_pdf_text(cv_bytes)
    if not cv_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the uploaded CV PDF")

    jobs = await fetch_all_jobs()
    ranked = await score_jobs_with_gemini(cv_text, role, jobs)
    return {"count": len(ranked), "jobs": ranked}
