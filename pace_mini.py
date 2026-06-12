#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACE MINI — drop this file into any folder and run it.
A fully-local AI workspace assistant. No cloud. No compilation. No clang.

    python pace_mini.py

Single file. Auto-downloads Ollama + the LFM2.5 model. Sandboxes every
file operation to the directory it lives in. Works on Windows (x64 + ARM64),
macOS (Intel + Apple Silicon), and Linux.
"""

import os
import sys
import re
import json
import stat
import zipfile
import tarfile
import subprocess
import platform
import shutil
import signal
import time
import atexit

# ---------------------------------------------------------------------------
# Resolve our own path early — works whether invoked as `python pace_mini.py`
# or `py pace_mini.py` or from a different cwd.
# ---------------------------------------------------------------------------
_SCRIPT_PATH = os.path.realpath(os.path.abspath(__file__))
if not os.path.isfile(_SCRIPT_PATH):
    _SCRIPT_PATH = os.path.join(os.getcwd(), "pace_mini.py")

ROOT        = os.path.dirname(_SCRIPT_PATH)
SCRIPT_NAME = os.path.basename(_SCRIPT_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_TAG   = "lfm2.5-350m"          # local ollama model name (created from GGUF)
MODEL_LABEL = "LFM2.5-350M"

# The GGUF file sits right next to the script — easy to see, easy to delete/replace
MODEL_GGUF_NAME = "LFM2.5-350M-Q4_K_M.gguf"
MODEL_GGUF_URL  = (
    "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF"
    "/resolve/main/LFM2.5-350M-Q4_K_M.gguf"
)
MODEL_GGUF_PATH = os.path.join(ROOT, MODEL_GGUF_NAME)

OLLAMA_DIR  = os.path.join(ROOT, ".ollama_engine")   # engine lives here
OLLAMA_DATA = os.path.join(ROOT, ".ollama_models")   # models live here

# Platform-specific Ollama binary download URLs
# We always grab the native binary for the running architecture.
def _ollama_asset():
    sys_  = platform.system()
    mach  = platform.machine().lower()
    arm   = mach in ("arm64", "aarch64", "armv8")
    if sys_ == "Windows":
        # Ollama ships a zip for Windows with both arches inside
        url  = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
        exe  = os.path.join(OLLAMA_DIR, "ollama.exe")
        kind = "zip"
        # ARM64 Windows: still grab amd64 zip — Windows ARM runs x64 via emulation,
        # and the ollama team now ships an arm64 exe inside the same zip.
        if arm:
            url  = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-arm64.zip"
    elif sys_ == "Darwin":
        # macOS: single zip, universal binary inside
        url  = "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.zip"
        exe  = os.path.join(OLLAMA_DIR, "ollama")
        kind = "zip"
    else:
        # Linux: tgz, amd64 or arm64
        arch = "arm64" if arm else "amd64"
        url  = f"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-{arch}.tgz"
        exe  = os.path.join(OLLAMA_DIR, "ollama")
        kind = "tgz"
    return url, exe, kind

OLLAMA_URL, OLLAMA_EXE, OLLAMA_ARCHIVE_KIND = _ollama_asset()
OLLAMA_PORT = 11435   # use a non-default port so we never clash with a user's existing Ollama
OLLAMA_BASE = f"http://127.0.0.1:{OLLAMA_PORT}"

CTX_TOKENS           = 80000
MAX_FILE_CONTEXT_BYTES = 50 * 1024
MAX_SYSTEM_PROMPT_CHARS = 9000
MAX_GEN_TOKENS       = 32500

# Palette
C_AI   = "#7EB8F7"   # electric blue  — AI responses
C_FILE = "#F5A623"   # amber          — file ops
C_ERR  = "#E8735A"   # coral          — errors
C_DIM  = "#555555"   # chrome/borders

SKIP_DIRS  = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
              ".ollama_engine", ".ollama_models"}
SKIP_FILES = {SCRIPT_NAME, MODEL_GGUF_NAME, MODEL_GGUF_NAME + ".part"}

# ---------------------------------------------------------------------------
# Minimal bootstrap print (rich not available yet)
# ---------------------------------------------------------------------------
def _p(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Auto-install pip dependencies (only rich + requests needed now)
# ---------------------------------------------------------------------------
def _have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False

def _pip(*pkgs):
    base = [sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check"]
    r = subprocess.run(base + list(pkgs))
    if r.returncode != 0:
        r = subprocess.run(base + ["--user"] + list(pkgs))
    return r.returncode == 0

missing = [p for p in ("rich", "requests") if not _have(p)]
if missing:
    _p(f"  Installing {', '.join(missing)} ...")
    if not _pip(*missing):
        _p(f"  ERROR: could not install {', '.join(missing)}.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Now safe to import everything
# ---------------------------------------------------------------------------
import requests
from rich.console import Console
from rich.tree    import Tree
from rich.text    import Text
from rich.panel   import Panel
from rich.syntax  import Syntax
from rich.rule    import Rule
from rich.progress import (
    Progress, BarColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn, TextColumn,
)

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Ollama process management
# ---------------------------------------------------------------------------
_ollama_proc = None   # the subprocess.Popen handle we own

def _stop_ollama():
    global _ollama_proc
    if _ollama_proc and _ollama_proc.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(_ollama_proc.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(_ollama_proc.pid), signal.SIGTERM)
        except Exception:
            try:
                _ollama_proc.terminate()
            except Exception:
                pass
        _ollama_proc = None

atexit.register(_stop_ollama)

def _signal_handler(sig, frame):
    _stop_ollama()
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Download helper (resumable, progress bar)
# ---------------------------------------------------------------------------
def _download(url, dest, label="downloading"):
    part = dest + ".part"
    resume = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "pace-mini/2.0"}
    if resume:
        headers["Range"] = f"bytes={resume}-"
        console.print(f"  [{C_DIM}]Resuming at {resume/1e6:.1f} MB...[/]")

    with requests.get(url, stream=True, headers=headers,
                      timeout=30, allow_redirects=True) as r:
        if resume and r.status_code == 200:
            resume = 0   # server ignored Range
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume

        progress = Progress(
            TextColumn(f"[{C_DIM}]{label}[/]"),
            BarColumn(bar_width=28, complete_style=C_AI, finished_style=C_AI),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        mode = "ab" if resume else "wb"
        with progress:
            task = progress.add_task("dl", total=total or None, completed=resume)
            with open(part, mode) as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    os.replace(part, dest)


# ---------------------------------------------------------------------------
# Stage 1 — ensure Ollama binary exists
# ---------------------------------------------------------------------------
def ensure_ollama():
    if os.path.isfile(OLLAMA_EXE):
        return

    os.makedirs(OLLAMA_DIR, exist_ok=True)
    archive = os.path.join(OLLAMA_DIR, "ollama_archive")
    console.print(f"  [{C_DIM}]Downloading Ollama engine...[/]")
    _download(OLLAMA_URL, archive, label="ollama")

    console.print(f"  [{C_DIM}]Extracting...[/]")
    if OLLAMA_ARCHIVE_KIND == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(OLLAMA_DIR)
    else:
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(OLLAMA_DIR)
    os.remove(archive)

    # Make executable on Unix
    if os.name != "nt":
        st = os.stat(OLLAMA_EXE)
        os.chmod(OLLAMA_EXE, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    if not os.path.isfile(OLLAMA_EXE):
        # Some archives nest inside a subdir — walk and find it
        for dirpath, _, files in os.walk(OLLAMA_DIR):
            for fname in files:
                if fname.lower() in ("ollama", "ollama.exe"):
                    found = os.path.join(dirpath, fname)
                    dest  = OLLAMA_EXE
                    if found != dest:
                        shutil.move(found, dest)
                    if os.name != "nt":
                        st = os.stat(dest)
                        os.chmod(dest, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                    break

    if not os.path.isfile(OLLAMA_EXE):
        console.print(f"  [{C_ERR}]✗ Could not find ollama binary after extraction.[/]")
        console.print(f"  [{C_DIM}]Contents of {OLLAMA_DIR}:[/]")
        for dp, ds, fs in os.walk(OLLAMA_DIR):
            for f in fs:
                console.print(f"    {os.path.relpath(os.path.join(dp,f), OLLAMA_DIR)}")
        sys.exit(1)

    console.print(f"  [{C_FILE}]✓ Ollama engine ready[/]")


# ---------------------------------------------------------------------------
# Stage 2 — start Ollama server
# ---------------------------------------------------------------------------
def start_ollama_server():
    global _ollama_proc

    env = dict(os.environ)
    env["OLLAMA_MODELS"] = OLLAMA_DATA        # keep models in our folder
    env["OLLAMA_HOST"]   = f"127.0.0.1:{OLLAMA_PORT}"
    os.makedirs(OLLAMA_DATA, exist_ok=True)

    kwargs = dict(
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.name != "nt":
        kwargs["start_new_session"] = True   # so Ctrl+C doesn't kill it directly

    _ollama_proc = subprocess.Popen([OLLAMA_EXE, "serve"], **kwargs)

    # Wait up to 15 s for the server to be ready
    with console.status(f"[{C_DIM}]Starting Ollama server...[/]",
                        spinner="dots", spinner_style=C_AI):
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=1)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.4)

    console.print(f"  [{C_ERR}]✗ Ollama server did not start in time.[/]")
    _stop_ollama()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 3a — download the GGUF file (resumable)
# ---------------------------------------------------------------------------
def ensure_gguf():
    if os.path.isfile(MODEL_GGUF_PATH):
        return
    console.print(f"  [{C_DIM}]Downloading {MODEL_GGUF_NAME} (~229 MB, one-time)...[/]")
    _download(MODEL_GGUF_URL, MODEL_GGUF_PATH, label="model")
    console.print(f"  [{C_FILE}]✓ model file saved[/]")


# ---------------------------------------------------------------------------
# Stage 3b — register the GGUF with our local Ollama instance
# ---------------------------------------------------------------------------
def _model_registered():
    """Return True if MODEL_TAG is already known to our Ollama server."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
        return any(m.get("name", "").split(":")[0] == MODEL_TAG
                   for m in tags)
    except Exception:
        return False


def ensure_model():
    if _model_registered():
        return

    # Write a temporary Modelfile that points to our GGUF on disk.
    # Ollama `create` reads FROM as an absolute path when it starts with /
    # or a drive letter on Windows.  We use the realpath to be safe.
    gguf_abs = os.path.realpath(MODEL_GGUF_PATH).replace("\\", "/")
    # On Windows Ollama needs the path in a FROM line — use forward slashes
    modelfile_content = "\n".join([
        f"FROM {gguf_abs}",
        "PARAMETER temperature 0.3",
        "PARAMETER repeat_penalty 1.1",
        f"PARAMETER num_predict {MAX_GEN_TOKENS}",
        "",
    ])
    modelfile_path = os.path.join(OLLAMA_DIR, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as fh:
        fh.write(modelfile_content)

    console.print(f"  [{C_DIM}]Registering {MODEL_LABEL} with Ollama (one-time)...[/]")

    env = dict(os.environ)
    env["OLLAMA_MODELS"] = OLLAMA_DATA
    env["OLLAMA_HOST"]   = f"127.0.0.1:{OLLAMA_PORT}"

    result = subprocess.run(
        [OLLAMA_EXE, "create", MODEL_TAG, "-f", modelfile_path],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        console.print(f"  [{C_ERR}]✗ Failed to register model:[/]")
        console.print(result.stderr or result.stdout)
        _stop_ollama()
        sys.exit(1)

    console.print(f"  [{C_FILE}]✓ {MODEL_LABEL} ready[/]")


# ---------------------------------------------------------------------------
# File-system sandbox
# ---------------------------------------------------------------------------
def safe_path(rel):
    if not rel or not isinstance(rel, str):
        return None
    rel = rel.strip().strip('"').strip("'")
    candidate = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    resolved  = os.path.realpath(candidate)
    if resolved == ROOT or resolved.startswith(ROOT + os.sep):
        return resolved
    return None

def is_text_file(path, blocksize=4096):
    try:
        with open(path, "rb") as f:
            block = f.read(blocksize)
        if b"\x00" in block:
            return False
        block.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False

def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} {unit}" if unit == "B" else f"{n/1024:.1f} {unit}"
        n /= 1024.0

def walk_workspace():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name in SKIP_FILES or name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            rel  = os.path.relpath(full, ROOT)
            try:
                out.append((rel, os.path.getsize(full)))
            except OSError:
                continue
    return out

def build_tree():
    tree  = Tree(f"[bold white]{os.path.basename(ROOT) or ROOT}/[/]",
                 guide_style=C_DIM)
    nodes = {"": tree}

    def node_for(dir_rel):
        if dir_rel in nodes:
            return nodes[dir_rel]
        parent = node_for(os.path.dirname(dir_rel))
        n = parent.add(f"[white]{os.path.basename(dir_rel)}/[/]")
        nodes[dir_rel] = n
        return n

    for rel, size in walk_workspace():
        parent = node_for(os.path.dirname(rel))
        name   = os.path.basename(rel)
        parent.add(Text.assemble((name, "white"), ("  " + human_size(size), C_DIM)))
    return tree


# ---------------------------------------------------------------------------
# AI context builder
# ---------------------------------------------------------------------------
SYSTEM_RULES = """\
You are Pace Mini, a local AI assistant for one folder on the user's computer.
You help with files in this folder only: answer questions, summarise content,
and create or edit files. You are also a fast assistant.

To create or overwrite a file output EXACTLY this block and nothing else around it:
WRITE_FILE: path/relative/to/root
<<<
file content here
>>>

Be concise. Short focused answers only."""

def build_system_prompt():
    files = walk_workspace()
    lines = [SYSTEM_RULES, "",
             f"WORKSPACE: {os.path.basename(ROOT) or ROOT}", "",
             "FILES (name — size):"]
    big, small = [], []
    for rel, size in files:
        lines.append(f"  {rel} — {human_size(size)}")
        (small if size <= MAX_FILE_CONTEXT_BYTES else big).append((rel, size))

    budget = MAX_SYSTEM_PROMPT_CHARS - sum(len(l) for l in lines)
    chunks = []
    for rel, size in small:
        full = os.path.join(ROOT, rel)
        if not is_text_file(full):
            continue
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_FILE_CONTEXT_BYTES)
        except OSError:
            continue
        block = f"\n--- FILE: {rel} ---\n{content}"
        block = block[:max(budget, 0)]
        if block:
            chunks.append(block)
            budget -= len(block)
        if budget <= 0:
            break

    if chunks:
        lines += ["", "FILE CONTENTS:"] + chunks
    if big:
        lines += ["", "LARGE FILES (use /read to load): " +
                  ", ".join(r for r, _ in big)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WRITE_FILE interception
# ---------------------------------------------------------------------------
WRITE_RE = re.compile(
    r"WRITE_FILE:\s*(?P<path>[^\n]+)\n<<<\n?(?P<body>.*?)\n?>>>",
    re.DOTALL,
)

def write_file_to_workspace(rel, content):
    resolved = safe_path(rel)
    if resolved is None:
        return False, f"refused: path escapes workspace ({rel})"
    if os.path.realpath(resolved) == os.path.realpath(_SCRIPT_PATH):
        return False, "refused: will not overwrite pace_mini.py"
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write(content)
    return True, os.path.relpath(resolved, ROOT)

def apply_write_blocks(text):
    wrote = 0
    for m in WRITE_RE.finditer(text):
        rel  = m.group("path").strip()
        body = m.group("body")
        ok, msg = write_file_to_workspace(
            rel, body if body.endswith("\n") else body + "\n")
        if ok:
            size = os.path.getsize(os.path.join(ROOT, msg))
            console.print(f"  [{C_FILE}]✓ wrote {msg} ({human_size(size)})[/]")
            wrote += 1
        else:
            console.print(f"  [{C_ERR}]✗ {msg}[/]")
    return wrote


# ---------------------------------------------------------------------------
# Chat — talks to Ollama's OpenAI-compatible /api/chat endpoint
# ---------------------------------------------------------------------------
class Chat:
    def __init__(self):
        self.system  = build_system_prompt()
        self.history = []   # list of {"role": ..., "content": ...}

    def reindex(self):
        self.system = build_system_prompt()

    def reset(self):
        self.history = []
        self.reindex()

    def _trim(self):
        if len(self.history) > 16:
            self.history = self.history[-16:]

    def ask(self, user_text):
        self._trim()
        messages = (
            [{"role": "system", "content": self.system}]
            + self.history
            + [{"role": "user", "content": user_text}]
        )
        reply = []
        console.print()
        try:
            with requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model":   MODEL_TAG,
                    "messages": messages,
                    "stream":   True,
                    "options": {
                        "num_predict": MAX_GEN_TOKENS,
                        "temperature": 0.3,
                        "repeat_penalty": 1.1,
                    },
                },
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code != 200:
                    console.print(f"  [{C_ERR}]✗ Ollama error {r.status_code}: {r.text[:200]}[/]")
                    return ""
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    tok = obj.get("message", {}).get("content", "")
                    if tok:
                        reply.append(tok)
                        console.print(Text(tok, style=C_AI), end="")
                    if obj.get("done"):
                        break
        except KeyboardInterrupt:
            console.print(f"\n  [{C_DIM}](interrupted)[/]")
        console.print()
        full = "".join(reply)
        self.history.append({"role": "user",      "content": user_text})
        self.history.append({"role": "assistant",  "content": full})
        return full


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WORDMARK = r"""
 ██████╗  █████╗  ██████╗███████╗   ███╗   ███╗██╗███╗   ██╗██╗
 ██╔══██╗██╔══██╗██╔════╝██╔════╝   ████╗ ████║██║████╗  ██║██║
 ██████╔╝███████║██║     █████╗     ██╔████╔██║██║██╔██╗ ██║██║
 ██╔═══╝ ██╔══██║██║     ██╔══╝     ██║╚██╔╝██║██║██║╚██╗██║██║
 ██║     ██║  ██║╚██████╗███████╗   ██║ ╚═╝ ██║██║██║ ╚████║██║
 ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝
""".rstrip("\n")

HELP_TEXT = f"""[white]/help[/]   [{C_DIM}]show this help[/]
[white]/ls[/]     [{C_DIM}]refresh the directory tree[/]
[white]/read[/]   [{C_DIM}]/read <path> — view a file with syntax highlighting[/]
[white]/write[/]  [{C_DIM}]/write <path> [content] — write a file (AI fills if no content)[/]
[white]/clear[/]  [{C_DIM}]reset conversation history (keeps workspace context)[/]
[white]/exit[/]   [{C_DIM}]quit  (Ctrl+C also works)[/]"""

def banner():
    console.print(Text(WORDMARK, style="bold white"))
    console.print(
        Text.assemble(
            ("  model ", C_DIM),  (MODEL_LABEL, "white"),
            ("   dir ",  C_DIM),  (ROOT, "white"),
        )
    )
    console.print(Rule(style=C_DIM))

def show_tree():
    console.print()
    console.print(build_tree())
    console.print()

def cmd_read(arg):
    resolved = safe_path(arg)
    if resolved is None or not os.path.isfile(resolved):
        console.print(f"  [{C_ERR}]✗ cannot read '{arg}'[/]")
        return
    if not is_text_file(resolved):
        console.print(f"  [{C_ERR}]✗ '{arg}' is binary[/]")
        return
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()
    rel   = os.path.relpath(resolved, ROOT)
    lexer = Syntax.guess_lexer(resolved, code)
    console.print(Panel(
        Syntax(code, lexer, theme="ansi_dark", line_numbers=True, word_wrap=True),
        title=f"[{C_FILE}]{rel}[/]", border_style=C_DIM, title_align="left",
    ))

def cmd_write(arg, chat):
    parts = arg.split(None, 1)
    if not parts:
        console.print(f"  [{C_ERR}]✗ usage: /write <path> [content][/]")
        return
    rel = parts[0]
    if len(parts) == 2:
        ok, msg = write_file_to_workspace(rel, parts[1] + "\n")
        if ok:
            size = os.path.getsize(os.path.join(ROOT, msg))
            console.print(f"  [{C_FILE}]✓ wrote {msg} ({human_size(size)})[/]")
            chat.reindex()
        else:
            console.print(f"  [{C_ERR}]✗ {msg}[/]")
    else:
        full = chat.ask(
            f"Create the file '{rel}'. Reply with ONLY a WRITE_FILE block "
            "containing complete, sensible content for that file."
        )
        if apply_write_blocks(full):
            chat.reindex()
        else:
            console.print(f"  [{C_ERR}]✗ model did not produce a WRITE_FILE block[/]")

def repl(chat):
    banner()
    show_tree()
    console.print(f"  [{C_DIM}]Ask anything — /help for commands[/]\n")

    while True:
        try:
            user = console.input(f"[bold white]pace ›[/] [{C_DIM}]▌[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{C_DIM}]bye.[/]")
            return
        user = user.strip()
        if not user:
            continue

        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("/exit", "/quit", "/q"):
                console.print(f"  [{C_DIM}]bye.[/]")
                return
            elif cmd == "/help":
                console.print(Panel(HELP_TEXT, border_style=C_DIM,
                                    title=f"[{C_DIM}]commands[/]",
                                    title_align="left"))
            elif cmd == "/ls":
                chat.reindex()
                show_tree()
            elif cmd == "/read":
                cmd_read(arg)
            elif cmd == "/write":
                cmd_write(arg, chat)
            elif cmd == "/clear":
                chat.reset()
                console.print(f"  [{C_DIM}]conversation cleared[/]")
            else:
                console.print(f"  [{C_ERR}]✗ unknown command — try /help[/]")
            continue

        full = chat.ask(user)
        if apply_write_blocks(full):
            chat.reindex()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if sys.version_info < (3, 8):
        print("pace mini needs Python 3.8+")
        sys.exit(1)

    t0 = time.time()

    # 1. Get the Ollama binary
    ensure_ollama()

    # 2. Start our private Ollama server
    start_ollama_server()

    # 3a. Download the GGUF file if needed
    ensure_gguf()
    # 3b. Register it with Ollama if needed
    ensure_model()

    # 4. Launch the REPL
    elapsed = time.time() - t0
    console.print(f"  [{C_DIM}]ready in {elapsed:.1f}s[/]")
    chat = Chat()
    repl(chat)

    # 5. Clean shutdown
    _stop_ollama()


if __name__ == "__main__":
    main()