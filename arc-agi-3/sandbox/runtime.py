from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

_SANDBOX = Path(__file__).resolve().parent


def _cache_dir() -> Path:
    return Path(os.environ.get("ARC_SANDBOX_CACHE", Path.home() / ".cache" / "arc-sandbox"))


def _ensure_enroot_image() -> Path:
    bake = _SANDBOX / "bake.sh"
    digest = hashlib.sha256(bake.read_bytes()).hexdigest()[:12]
    img = _cache_dir() / f"arc-agent-{digest}.sqsh"
    if img.is_file():
        return img
    print(f"[sandbox] image not cached, baking (needs network + a few minutes): {img}", flush=True)
    r = subprocess.run(["bash", str(bake)], stdout=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit(f"bake.sh failed with exit code {r.returncode}")
    out = Path(r.stdout.strip().splitlines()[-1])
    if not out.is_file():
        raise SystemExit(f"bake.sh reported {out}, but it does not exist")
    return out


def make_runtime(name: str, *, enroot_env: "dict | None" = None) -> "ContainerRuntime":
    if name == "docker":
        return DockerRuntime()
    if name == "podman":
        return PodmanRuntime()
    if name == "enroot":
        return EnrootRuntime(enroot_env=enroot_env or dict(os.environ))
    raise ValueError(f"unknown sandbox runtime: {name!r} (expected enroot|podman|docker)")


class ContainerRuntime:

    name = "container"
    proxy_in_container = False
    container_home = "/root"
    needs_cwd_tmpfs = False

    @property
    def popen_env(self) -> dict:
        raise NotImplementedError

    def ensure_image(self) -> str:
        raise NotImplementedError

    def create(self, container: str) -> None:
        raise NotImplementedError

    def cleanup(self, container: str) -> None:
        raise NotImplementedError

    def container_cmd(self, *, name: str, mounts: "list[tuple]", env: dict,
                      argv: "list[str]", network: "str | None" = None,
                      tmpfs: "list[str] | None" = None) -> "list[str]":
        raise NotImplementedError

    @staticmethod
    def record_ro(paths: "list[str]") -> "list[tuple]":
        return [(p, p, "ro") for p in paths if os.path.exists(p)]


class EnrootRuntime(ContainerRuntime):

    name = "enroot"

    def __init__(self, *, enroot_env: dict) -> None:
        self.enroot_env = enroot_env
        self._image: "str | None" = None

    @property
    def popen_env(self) -> dict:
        return self.enroot_env

    def ensure_image(self) -> str:
        self._image = str(_ensure_enroot_image())
        return self._image

    def create(self, container: str) -> None:
        r = subprocess.run(["enroot", "create", "-n", container, str(self._image)],
                           env=self.enroot_env, capture_output=True, text=True)
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"enroot create {container} failed: {r.stderr.strip()[:300]}")

    def cleanup(self, container: str) -> None:
        subprocess.run(["enroot", "remove", "-f", container],
                       env=self.enroot_env, capture_output=True)

    def container_cmd(self, *, name: str, mounts: "list[tuple]", env: dict,
                      argv: "list[str]", network: "str | None" = None,
                      tmpfs: "list[str] | None" = None) -> "list[str]":
        inner = ["enroot", "start", "--rw"]
        for src, dst, mode in mounts:
            xc = "x-create=file" if os.path.isfile(src) else "x-create=dir"
            inner += ["--mount", f"{src}:{dst}:{mode},rbind,{xc}"]
        for k, v in env.items():
            inner += ["--env", f"{k}={v}"]
        inner += [name, *argv]
        return ["unshare", "--user", "--map-root-user", "--net", "--",
                "bash", "-c", 'ip link set lo up && exec "$@"', "netns", *inner]


class _OciCliRuntime(ContainerRuntime):

    _IMAGE_TAG = "arc-agent-sandbox:latest"

    def __init__(self) -> None:
        self._image = self._IMAGE_TAG

    def cli(self, *args: str) -> "list[str]":
        return [self.name, *args]

    def _run(self, *args: str, **kw):
        return subprocess.run(self.cli(*args), env=self.popen_env, **kw)

    def _identity_args(self) -> "list[str]":
        return []

    @property
    def popen_env(self) -> dict:
        return dict(os.environ)

    def _image_present(self) -> bool:
        return self._run("image", "inspect", self._image, capture_output=True).returncode == 0

    def _build_image(self) -> None:
        df = _SANDBOX / "Dockerfile"
        print(f"[sandbox] {self.name}: building {self._image} from {df} "
              f"(needs network + a few minutes)", flush=True)
        r = self._run("build", "-t", self._image, "-f", str(df), str(_SANDBOX),
                      capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{self.name} build failed: {r.stderr.strip()[-400:]}")

    def ensure_image(self) -> str:
        if not self._image_present():
            self._build_image()
        return self._image

    def create(self, container: str) -> None:
        return None

    def cleanup(self, container: str) -> None:
        self._run("rm", "-f", container, capture_output=True)

    def network_create(self, name: str) -> None:
        r = self._run("network", "create", "--internal", name, capture_output=True, text=True)
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"{self.name} network create {name} failed: {r.stderr.strip()[:300]}")

    def network_rm(self, name: str) -> None:
        self._run("network", "rm", name, capture_output=True)

    def container_cmd(self, *, name: str, mounts: "list[tuple]", env: dict,
                      argv: "list[str]", network: "str | None" = None,
                      tmpfs: "list[str] | None" = None) -> "list[str]":
        cmd = self.cli("run", "-i", "--rm", "--name", name, *self._identity_args())
        cmd += ["--network", network or "none"]
        for p in tmpfs or []:
            cmd += ["--tmpfs", p]
        for src, dst, mode in mounts:
            cmd += ["-v", f"{src}:{dst}" + (":ro" if mode == "ro" else "")]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self._image, *argv]
        return cmd


class DockerRuntime(_OciCliRuntime):

    name = "docker"
    proxy_in_container = True
    container_home = "/tmp"
    needs_cwd_tmpfs = True

    def _identity_args(self) -> "list[str]":
        return ["--user", f"{os.getuid()}:{os.getgid()}"]


class PodmanRuntime(_OciCliRuntime):

    name = "podman"

    def __init__(self) -> None:
        super().__init__()
        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        root = Path(os.environ.get("ARC_SANDBOX_PODMAN_ROOT", tmp / "arc-podman"))
        self._graphroot = root / "storage"
        self._runroot = root / "run"
        for d in (self._graphroot, self._runroot):
            d.mkdir(parents=True, exist_ok=True)
        self._env = dict(os.environ, XDG_RUNTIME_DIR=str(self._runroot))
        wrapper = _SANDBOX / "crun-nokeyring"
        self._runtime = (str(wrapper) if wrapper.is_file() and os.access(wrapper, os.X_OK)
                         and shutil.which("crun") else None)

    def cli(self, *args: str) -> "list[str]":
        cmd = ["podman", "--root", str(self._graphroot), "--runroot", str(self._runroot)]
        if self._runtime:
            cmd += ["--runtime", self._runtime]
        return [*cmd, *args]

    @property
    def popen_env(self) -> dict:
        return self._env

    def _image_tar(self) -> Path:
        digest = hashlib.sha256((_SANDBOX / "Dockerfile").read_bytes()).hexdigest()[:12]
        return _cache_dir() / f"arc-agent-{digest}.tar"

    def ensure_image(self) -> str:
        if self._image_present():
            return self._image
        tar = self._image_tar()
        if tar.is_file():
            print(f"[sandbox] podman: loading cached image {tar}", flush=True)
            r = self._run("load", "-i", str(tar), capture_output=True, text=True)
            if r.returncode == 0 and self._image_present():
                return self._image
            print(f"[sandbox] podman: load failed, rebuilding ({(r.stderr or '').strip()[:200]})",
                  flush=True)
        self._build_image()
        tmp_tar = tar.with_suffix(f".tmp.{os.getpid()}")
        tar.parent.mkdir(parents=True, exist_ok=True)
        r = self._run("save", "-o", str(tmp_tar), self._image, capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp_tar, tar)
            print(f"[sandbox] podman: image cached at {tar}", flush=True)
        else:
            tmp_tar.unlink(missing_ok=True)
        return self._image
