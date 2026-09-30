# Repository Guidelines

## Project Structure & Architecture

- `app.py` is the Flask entry point and coordinates jobs for transcription, summarization, TTS, image generation, and OCR.
- Supporting pipeline modules live at the repository root: `audio_utils.py`, `transcribe_*.py`, `summarize.py`, `tts.py`, `zimage.py`, and `ocr.py`.
- `load_env.py` is the central configuration loader. Web pages are in `templates/`.
- Runtime data is written to ignored directories such as `uploads/`, `temp_segments/`, `results/`, `tts_output/`, `zimage_output/`, `ocr_upload/`, and `logs/`. Never commit recordings, transcripts, generated media, or models.

## Build, Test, and Development Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py                         # Flask UI at http://localhost:5001
python -m compileall *.py             # syntax smoke check
bash start_all.sh                      # configured host: llama server + Flask
bash check_srv.sh                      # check ports 8080–8082 and 5001
bash stop_all.sh                       # stop local services
```

Install `ffmpeg` separately. `start_all.sh` contains host-specific absolute paths; use it only where the referenced binaries and models exist. Prefer `.env` for configurable paths.

## Coding Style & Naming Conventions

Use Python 3.13+, four-space indentation, `snake_case` functions/variables, `PascalCase` classes, and type hints for public helpers. Centralize configuration in `load_env.py`; never hardcode secrets or machine-local paths. Preserve Vietnamese user-facing text and concise module docstrings. No formatter or linter is configured.

## Testing Guidelines

There is no checked-in automated test suite or coverage threshold. For every change, run `python -m compileall *.py`, start the app, and exercise the affected page or API route. Verify ffmpeg, Ollama/llama.cpp, and model binaries separately when relevant. If adding tests, use `tests/test_*.py` and document new dependencies.

## Commit & Pull Request Guidelines

Existing history uses short, informal subjects and has no enforced convention. Use a concise imperative subject with a conventional prefix where appropriate, such as `fix: handle empty audio uploads` or `docs: update local setup`. Keep commits focused. Pull requests should describe behavior changes, list validation commands and required local services/models, link related issues, and include screenshots for template/UI changes. Exclude `.env`, credentials, and generated media.

## Security & Configuration

Keep secrets and local model locations in the ignored `.env` file. Review upload handling and subprocess changes carefully: uploaded files can be large, and model commands execute locally with filesystem access. Avoid expanding allowed file types or service exposure without documenting the security impact.
