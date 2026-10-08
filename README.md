# Career Fit Multi-Agent Assistant

Capstone project for AAI-6270 Agentic AI, Lebanese American University (LAU).
Maha Hazime-Zayour

A multi-agent assistant that helps students and early-career job seekers apply for jobs
without exaggerating. From a CV and a job posting (or a search request) it produces a fit
analysis, a tailored resume, a cover letter, or a list of live remote jobs. Its main rule:
every claim must be supported by the CV.

## How it works

- **8 agents** (LangChain + LangGraph, local Ollama models): Supervisor, Candidate Evidence,
  Job Requirements, Fit & Gap, Resume Tailoring, Cover Letter, Job Search Query,
  Truthfulness Validator.
- **Coordination:** the Supervisor routes each request to a fixed LangGraph pipeline
  (sequential handoffs through a shared state). The writers and the Validator form a
  producer/critic repair loop (at most 2 repairs; a document that still fails is not shown).
- **Tools:** a CV reader (PDF, DOCX, TXT) and a live job search exposed through a local
  **MCP server**. The Job Search Query agent discovers the `search_jobs` tool over MCP and
  calls it through function calling; Python checks the arguments.
- **Memory:** short-term only (LangGraph shared state and a MemorySaver checkpoint for the
  pause while the user picks a job). No long-term memory or RAG.
- **Safeguards:** size limits and an AI input guard that runs first and fails closed; untrusted
  text wrapped in tags; safety rules in every prompt; the Validator on a second model to reduce
  self-preference bias; the user chooses the job, and nothing is ever sent or submitted.
- **Monitoring:** a run trace after each run, a run log in `logs/runs.jsonl` (never CV text),
  and a drift check.

## Workflows

Every request first passes the input guard. The Supervisor then picks one of seven workflows:

| Request | Agents in order |
|---|---|
| Fit analysis | Candidate Evidence → Job Requirements → Fit & Gap |
| Resume tailoring | Candidate Evidence → Job Requirements → Fit & Gap → Resume Tailoring → Validator |
| Cover letter | Candidate Evidence → Job Requirements → Fit & Gap → Cover Letter → Validator |
| Fit analysis and resume tailoring | Same as resume tailoring, with the gaps also shown |
| Keyword job search | Job Search Query → MCP job search |
| Job search based on the CV | Candidate Evidence → Job Search Query → MCP job search |
| Job search, then a cover letter | Candidate Evidence → Job Search Query → MCP job search → user picks a job → Job Requirements → Fit & Gap → Cover Letter → Validator |

Requests that are not about careers are refused before any CV is asked for.

## Requirements

- Python 3.10 or newer (tested with 3.12)
- [Ollama](https://ollama.com) running locally, with about 12 GB free disk space for the two models
- Internet access for the job search only (the CV never leaves the computer)
- No API keys are needed

## Setup

    git clone https://github.com/Maha-HazimeZayour/career-fit-multi-agent-assistant.git
    cd career-fit-multi-agent-assistant
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ollama pull gemma4:e4b
    ollama pull qwen2.5:7b
    cp .env.example .env

On Windows, activate the environment with `.venv\Scripts\activate`.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:e4b` | Model for all agents except the Validator |
| `OLLAMA_VALIDATOR_MODEL` | `qwen2.5:7b` | Model for the Truthfulness Validator |
| `OLLAMA_NUM_CTX` | `12288` | Context window size |
| `OLLAMA_NUM_PREDICT` | `3072` | Maximum length of an answer |
| `MAX_RETRIES` | `2` | Validator repair attempts |

## Run manually

Start Ollama, then:

    python app.py

Type a request, then the file paths the program asks for. Examples with the sample files:

| What you want | Request to type | Then type |
|---|---|---|
| Fit analysis | `Analyze how well my CV fits this job and show the main gaps.` | `samples/cv/cv_data_analyst_layla_haddad.docx`, then `samples/jobs/job_data_analyst_junior_en.txt` |
| Tailored resume | `Tailor my resume for this job.` | `samples/cv/sample_cv_healthcare.docx`, then `samples/jobs/sample_job_healthcare.txt` |
| Cover letter | `Write a cover letter for this job.` | `samples/cv/cv_accountant_lukas_brenner.pdf`, then `samples/jobs/job_accountant_de.txt` |
| Keyword search | `Find junior data analyst jobs in Germany.` | nothing |
| Search from the CV | `Find jobs based on my CV.` | `samples/cv/cv_backend_omar_alkhatib.txt` |
| Search, choose, letter | `Find jobs based on my CV and write a cover letter for one.` | `samples/cv/cv_data_analyst_layla_haddad.docx`, then a job number when the list appears |

For the job posting you can also press Enter and paste the text, then type `END` on its own line.
A run takes about 1 to 5 minutes on a laptop.

## Tests

**Automated tests** (no Ollama needed, a fake model replaces it, a few seconds):

    pytest -q

Expected: `106 passed`.

**Scenario tests with the real models** (23 fixed inputs with expected results; outputs saved in `results/`):

    bash run_tests.sh
    bash run_tests.sh T12 T15 T18

The first runs all tests (about 1 hour), the second only the ones named.

| Tests | What they cover |
|---|---|
| T1 | Out-of-scope request is refused |
| T2, T11, T21, T23 | Keyword search, level and employment-type filters, country filter |
| T3, T12 | Fit analysis and gaps |
| T4, T9, T13, T17 | Resume tailoring and full flow; T13 asks to exaggerate |
| T5, T14, T15, T16 | Cover letters in English, from an Italian CV, in Arabic and in German |
| T6, T19 | Job search based on the CV |
| T7, T20 | Search, pause, job choice, cover letter |
| T8 | Request to exaggerate a basic skill |
| T10, T18 | Hidden instructions inside a job posting (prompt injection) |
| T22 | Wrong file given as the job posting |

The outputs of the final runs (`T1.txt` to `T23.txt`) and their run log (`runs.jsonl`) are in the
`evaluation/` folder.

**Drift check** (after at least 20 runs):

    python monitoring.py

## Project structure

    app.py, cli_helpers.py       command line entry, input reading, graph runs
    graph.py, workflow_routes.py LangGraph workflow and routing
    state.py                     shared state
    agents/                      the 8 agents and shared helpers
    workflow_nodes/              repair counter, job search, job choice, final responses
    tools/                       CV reader, Himalayas API call, MCP client
    mcp_job_server.py            MCP server with the search_jobs tool
    guardrails.py                input guard, safety rules, tag wrapping
    monitoring.py                run trace, run log, drift check
    config.py                    models and limits
    samples/                     sample CVs and job postings (all people and companies are fictional)
    tests/                       automated tests
    run_tests.sh                 scenario tests
    evaluation/                  outputs and run log of the final scenario tests

## Limitations

- The Validator is a small local model and can miss unsupported wording (for example
  "skilled" or a duty taken from the job posting), so documents should still be read by a person.
- Prompt injection was tested only in English and only inside job postings. Hidden text in a CV
  (for example white text in a Word file) or in another language may not be caught.
- The job source lists remote jobs only and limits how many requests can be made.
- Scanned CVs cannot be read (no OCR).
- In T15, the Arabic letter kept the greeting and closing in English.
- Each scenario test ran once; the results show the system works on these cases, not a measured accuracy.

## Data source

Live jobs come from the free [Himalayas](https://himalayas.app) jobs API. Results link back to
the original listings on Himalayas, as its terms require.
