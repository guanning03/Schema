from __future__ import annotations

import ctypes
import os
import sys

_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_REFER, _TRUNCATE, _IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_V1_ALL = (1 << 13) - 1
_FILE_ONLY = _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE | _IOCTL_DEV
_READONLY = _EXECUTE | _READ_FILE | _READ_DIR


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def apply(deny_root: "str | None", rw_dir: str, ro_dir: "str | list[str]") -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(_SYS_CREATE, None, 0, 1)
    if abi < 1:
        raise OSError("Landlock unavailable")
    handled = (_V1_ALL | (_REFER if abi >= 2 else 0) | (_TRUNCATE if abi >= 3 else 0)
               | (_IOCTL_DEV if abi >= 5 else 0))
    attr = _RulesetAttr(handled)
    rs_fd = libc.syscall(_SYS_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if rs_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")

    def allow(path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not os.path.isdir(path):
                access &= _FILE_ONLY
            pb = _PathBeneath(access & handled, fd)
            if libc.syscall(_SYS_ADD_RULE, rs_fd, _RULE_PATH_BENEATH, ctypes.byref(pb), 0):
                raise OSError(ctypes.get_errno(), f"landlock_add_rule({path})")
        finally:
            os.close(fd)

    if deny_root is not None:
        deny = os.path.realpath(deny_root)
        cur, parts = "/", [p for p in deny.split("/") if p]
        for nxt in parts:
            for entry in os.listdir(cur):
                if entry == nxt:
                    continue
                p = os.path.join(cur, entry)
                r = os.path.realpath(p).rstrip("/") or "/"
                if deny == r or deny.startswith(r + "/") or r.startswith(deny + "/"):
                    continue
                allow(p, handled)
            cur = os.path.join(cur, nxt)
    allow(os.path.realpath(rw_dir), handled)
    ro_dirs = [ro_dir] if isinstance(ro_dir, str) else ro_dir
    for path in ro_dirs:
        allow(os.path.realpath(path), _READONLY)
    if deny_root is None:
        for dev in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"):
            allow(dev, handled)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
    if libc.syscall(_SYS_RESTRICT, rs_fd, 0):
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(rs_fd)


def main() -> None:
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
    except ValueError:
        sep = -1
    allow_only = bool(argv and argv[0] == "--allow-only")
    if sep < 3 or sep + 1 >= len(argv):
        print("usage: landlock_exec.py DENY_ROOT RW_DIR RO_DIR... -- CMD...\n"
              "   or: landlock_exec.py --allow-only RW_DIR RO_DIR... -- CMD...",
              file=sys.stderr)
        sys.exit(2)
    if allow_only:
        deny, rw, ro = None, argv[1], argv[2:sep]
    else:
        deny, rw, ro = argv[0], argv[1], argv[2:sep]
    cmd = argv[sep + 1:]
    try:
        apply(deny, rw, ro)
    except OSError as e:
        print(f"landlock_exec: FAILED to sandbox: {e}", file=sys.stderr)
        sys.exit(97)
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        print(f"landlock_exec: cannot exec {cmd[0]}: {e}", file=sys.stderr)
        sys.exit(126)


if __name__ == "__main__":
    main()
