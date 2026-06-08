from __future__ import annotations

from pathlib import Path
import re
import runpy
import shutil
import sys
import tempfile
import types


def _install_optional_dependency_stubs() -> None:
    if "rembg" not in sys.modules:
        rembg_stub = types.ModuleType("rembg")

        def _missing_rembg(*_args, **_kwargs):
            raise RuntimeError("rembg is unavailable in headless T2V mode.")

        rembg_stub.remove = _missing_rembg
        rembg_stub.new_session = lambda *_args, **_kwargs: None
        sys.modules["rembg"] = rembg_stub


def _build_patched_script(repo_dir: Path) -> Path:
    src = repo_dir / "wgp.py"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    temp_dir = Path(tempfile.mkdtemp(prefix="wan_headless_"))
    dst = temp_dir / "wgp_headless_runtime.py"
    shutil.copy2(src, dst)
    lines = dst.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    def find_validate_settings_insertion(lines_: list[str]) -> tuple[int | None, int | None]:
        for i, line in enumerate(lines_):
            if line.startswith("def validate_settings("):
                j = i + 1
                while j < len(lines_) and lines_[j].strip() == "":
                    j += 1
                if j < len(lines_) and lines_[j].lstrip().startswith(('"""', "'''")):
                    quote = '"""' if '"""' in lines_[j] else "'''"
                    j += 1
                    while j < len(lines_) and quote not in lines_[j]:
                        j += 1
                    if j < len(lines_):
                        j += 1
                return i, j
        return None, None

    marker = "HEADLESS FIX: avoid KeyError on inputs[...] in validate_settings"
    _, insertion_index = find_validate_settings_insertion(lines)
    if insertion_index is not None and not any(marker in line for line in lines):
        indent = " " * 4
        patch = [
            indent + "# --- HEADLESS FIX: avoid KeyError on inputs[...] in validate_settings ---\n",
            indent + "class _Inputs(dict):\n",
            indent + "    def __missing__(self, k):\n",
            indent + "        try:\n",
            indent + "            return primary_settings.get(k, None)\n",
            indent + "        except Exception:\n",
            indent + "            return None\n",
            indent + "inputs = _Inputs(inputs or {})\n",
            indent + "# --- END HEADLESS FIX ---\n\n",
        ]
        lines = lines[:insertion_index] + patch + lines[insertion_index:]

    text = "".join(lines)
    pattern = (
        r"(?P<indent>[ \t]*)(?:success\s*=\s*)?generate_video\(\s*task\s*,\s*send_cmd\s*,\s*plugin_data\s*=\s*plugin_data\s*,"
        r"\s*\*\*filtered_params\s*\)"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}# --- HEADLESS FIX: inject required generate_video() args if missing ---\n"
            f"{indent}_params = task.get('params') if isinstance(task, dict) else None\n"
            f"{indent}if not isinstance(_params, dict):\n"
            f"{indent}    _params = {{}}\n"
            f"{indent}_video_length = filtered_params.get('video_length', _params.get('video_length', primary_settings.get('video_length', 49)))\n"
            f"{indent}try:\n"
            f"{indent}    _video_length = int(_video_length)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _video_length = 49\n"
            f"{indent}_sw_size = filtered_params.pop('sliding_window_size', _params.get('sliding_window_size', primary_settings.get('sliding_window_size', min(_video_length, 24))))\n"
            f"{indent}try:\n"
            f"{indent}    _sw_size = int(_sw_size)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _sw_size = min(_video_length, 24)\n"
            f"{indent}_sw_overlap = filtered_params.pop('sliding_window_overlap', _params.get('sliding_window_overlap', primary_settings.get('sliding_window_overlap', max(1, _sw_size // 2))))\n"
            f"{indent}try:\n"
            f"{indent}    _sw_overlap = int(_sw_overlap)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _sw_overlap = max(1, _sw_size // 2)\n"
            f"{indent}if _sw_overlap < 1:\n"
            f"{indent}    _sw_overlap = 1\n"
            f"{indent}_sw_overlap_noise = filtered_params.pop('sliding_window_overlap_noise', _params.get('sliding_window_overlap_noise', primary_settings.get('sliding_window_overlap_noise', 0.0)))\n"
            f"{indent}try:\n"
            f"{indent}    _sw_overlap_noise = float(_sw_overlap_noise)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _sw_overlap_noise = 0.0\n"
            f"{indent}_sw_discard = filtered_params.pop('sliding_window_discard_last_frames', _params.get('sliding_window_discard_last_frames', primary_settings.get('sliding_window_discard_last_frames', 0)))\n"
            f"{indent}try:\n"
            f"{indent}    _sw_discard = int(_sw_discard)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _sw_discard = 0\n"
            f"{indent}_mode = filtered_params.pop('mode', _params.get('mode', primary_settings.get('mode', _params.get('model_type', 't2v'))))\n"
            f"{indent}if _mode is None:\n"
            f"{indent}    _mode = 't2v'\n"
            f"{indent}generate_video(\n"
            f"{indent}    task, send_cmd,\n"
            f"{indent}    plugin_data=plugin_data,\n"
            f"{indent}    sliding_window_size=_sw_size,\n"
            f"{indent}    sliding_window_overlap=_sw_overlap,\n"
            f"{indent}    sliding_window_overlap_noise=_sw_overlap_noise,\n"
            f"{indent}    sliding_window_discard_last_frames=_sw_discard,\n"
            f"{indent}    mode=_mode,\n"
            f"{indent}    **filtered_params\n"
            f"{indent})\n"
            f"{indent}# --- END HEADLESS FIX ---"
        )

    text, replacements = re.subn(pattern, repl, text)
    if replacements == 0:
        raise RuntimeError("Could not patch generate_video(...) call in wgp.py")

    dst.write_text(text, encoding="utf-8")
    return dst


def main() -> None:
    _install_optional_dependency_stubs()
    repo_dir = Path(__file__).resolve().parent
    patched_script = _build_patched_script(repo_dir)
    sys.argv = [str(patched_script), *sys.argv[1:]]
    runpy.run_path(str(patched_script), run_name="__main__")


if __name__ == "__main__":
    main()
