#!/usr/bin/env python3
"""
gta_solid.py - pinta TODAS as texturas do GTA SA (PC/D3D9) com uma cor lisa (bege),
visual de "maquete": so a luz e a sombra mostram as formas.

Le:    todos os .img e .txd soltos do jogo
Grava: <jogo>/modloader/solid_mode/...  (nao mexe nos originais)

Uso:
    pip install numpy
    python3 gta_solid.py .
    python3 gta_solid.py . --color CDAA7D

Mantem formato, tamanho, mipmaps e transparencia (alpha) de cada textura;
so troca a cor.
"""
import argparse
import os
import struct
import sys

import numpy as np

SECTOR = 2048
RW_STRUCT, RW_TEXNATIVE, RW_TXD = 0x01, 0x15, 0x16
DXT1, DXT3, DXT5 = (struct.unpack("<I", s)[0] for s in (b"DXT1", b"DXT3", b"DXT5"))


def chunk(buf, pos):
    t, s, v = struct.unpack_from("<III", buf, pos)
    return t, s, v, pos + 12


def mk(t, body, ver):
    return struct.pack("<III", t, len(body), ver) + body


# ---------------------------------------------------------------- cor lisa, mesmo formato
def paint(fmt, data, rgb):
    """Troca a cor de um nivel de pixels mantendo formato, tamanho e alpha."""
    r, g, b = rgb
    c565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    c555 = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    c444 = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
    luma = int(0.299 * r + 0.587 * g + 0.114 * b)
    a = np.frombuffer(data, np.uint8).copy()

    if fmt == DXT1:  # c0 = c1 = cor; pixel transparente continua transparente
        blk = a[:len(a) // 8 * 8].reshape(-1, 8)
        c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        idx = blk[:, 4:8].copy().view("<u4")[:, 0]
        ind = (idx[:, None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3
        transp = (c0 <= c1)[:, None] & (ind == 3)
        new = (np.where(transp, 3, 0).astype(np.uint32) << (np.arange(16, dtype=np.uint32) * 2)).sum(1, dtype=np.uint32)
        blk[:, 0:4] = np.frombuffer(struct.pack("<HH", c565, c565), np.uint8)
        blk[:, 4:8] = new.astype("<u4").view(np.uint8).reshape(-1, 4)
    elif fmt in (DXT3, DXT5):  # alpha (8 primeiros bytes) intacto, so o bloco de cor muda
        blk = a[:len(a) // 16 * 16].reshape(-1, 16)
        blk[:, 8:12] = np.frombuffer(struct.pack("<HH", c565, c565), np.uint8)
        blk[:, 12:16] = 0
    elif fmt in (21, 22):  # A8R8G8B8 / X8R8G8B8 (BGRA na memoria)
        px = a[:len(a) // 4 * 4].reshape(-1, 4)
        px[:, 0], px[:, 1], px[:, 2] = b, g, r
        if fmt == 22:
            px[:, 3] = 255
    elif fmt == 20:  # R8G8B8
        px = a[:len(a) // 3 * 3].reshape(-1, 3)
        px[:, 0], px[:, 1], px[:, 2] = b, g, r
    elif fmt in (23, 24, 25, 26, 51):  # formatos de 16 bits
        v = a[:len(a) // 2 * 2].view("<u2")
        if fmt == 23:
            v[:] = c565
        elif fmt == 24:
            v[:] = 0x8000 | c555
        elif fmt == 25:
            v[:] = (v & 0x8000) | c555
        elif fmt == 26:
            v[:] = (v & 0xF000) | c444
        else:  # A8L8
            v[:] = (v & 0xFF00) | luma
    elif fmt == 50:  # L8
        a[:] = luma
    elif fmt == 28:  # A8 (so alpha, nao tem cor)
        pass
    else:
        return None
    return a.tobytes()


def convert_native(body, rgb):
    t, s, sv, p = chunk(body, 0)
    st, tail = body[p:p + s], body[p + s:]
    if struct.unpack_from("<I", st, 0)[0] != 9:
        return body, False

    rfmt, fmt = struct.unpack_from("<II", st, 72)
    levels = st[85]
    out = bytearray(st[:88])
    off = 88

    pal_n = 256 if rfmt & 0x2000 else 32 if rfmt & 0x4000 else 0
    if pal_n:  # paletizada: so troca a cor da paleta, mantem alpha e indices
        pal = np.frombuffer(st, np.uint8, pal_n * 4, off).copy().reshape(pal_n, 4)
        pal[:, 0:3] = rgb
        out += pal.tobytes()
        off += pal_n * 4

    for _ in range(max(1, levels)):
        if off + 4 > len(st):
            break
        size = struct.unpack_from("<I", st, off)[0]
        data = st[off + 4:off + 4 + size]
        px = data if pal_n else paint(fmt, data, rgb)
        if px is None:
            return body, False  # formato desconhecido: mantem o original
        out += struct.pack("<I", size) + px
        off += 4 + size
    out += st[off:]
    return mk(RW_STRUCT, bytes(out), sv) + tail, True


def process_txd(buf, rgb):
    t, s, ver, p = chunk(buf, 0)
    if t != RW_TXD:
        raise ValueError("nao e um TXD")
    end = p + s
    t, s, v, q = chunk(buf, p)
    head = buf[p:q + s]
    p = q + s
    parts, n = [], 0
    while p < end:
        t, s, v, q = chunk(buf, p)
        if t == RW_TEXNATIVE:
            body, ok = convert_native(buf[q:q + s], rgb)
            parts.append(mk(RW_TEXNATIVE, body, v))
            n += ok
        else:
            parts.append(buf[p:q + s])
        p = q + s
    return mk(RW_TXD, head + b"".join(parts), ver), n


# ---------------------------------------------------------------- arquivos
def iter_img(path):
    with open(path, "rb") as f:
        if f.read(4) != b"VER2":
            print(f"  [pulando] {path} nao e IMG v2")
            return
        count = struct.unpack("<I", f.read(4))[0]
        entries = [struct.unpack("<IHH24s", f.read(32)) for _ in range(count)]
        for off, ss, asz, name in entries:
            name = name.split(b"\0")[0].decode("latin1")
            if name.lower().endswith(".txd"):
                f.seek(off * SECTOR)
                yield name, f.read((ss or asz) * SECTOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", help="pasta do GTA San Andreas")
    ap.add_argument("--color", default="CDAA7D", help="cor em hex (padrao: bege CDAA7D)")
    ap.add_argument("--out", default=None, help="padrao: <jogo>/modloader/solid_mode")
    a = ap.parse_args()

    rgb = tuple(int(a.color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    out_root = a.out or os.path.join(a.game, "modloader", "solid_mode")
    skip = os.path.normcase(os.path.join(a.game, "modloader"))
    stats = {"txd": 0, "tex": 0, "err": 0}

    def handle(name, data, dest_dir):
        try:
            new, n = process_txd(data, rgb)
        except Exception as e:
            stats["err"] += 1
            print(f"  [erro] {name}: {e}")
            return
        os.makedirs(dest_dir, exist_ok=True)
        with open(os.path.join(dest_dir, name), "wb") as f:
            f.write(new)
        stats["txd"] += 1
        stats["tex"] += n

    for root, dirs, files in os.walk(a.game):
        if os.path.normcase(root).startswith(skip):
            dirs[:] = []
            continue
        for fn in files:
            full = os.path.join(root, fn)
            low = fn.lower()
            if low.endswith(".img"):
                print(f"IMG {full}")
                dest = os.path.join(out_root, os.path.splitext(fn)[0])
                for name, data in iter_img(full):
                    handle(name, data, dest)
            elif low.endswith(".txd"):
                rel = os.path.relpath(root, a.game)
                with open(full, "rb") as f:
                    handle(fn, f.read(), os.path.join(out_root, "loose", rel))

    print(f"\nPronto: {stats['txd']} TXDs, {stats['tex']} texturas, {stats['err']} erros")
    print(f"Saida: {out_root}")


if __name__ == "__main__":
    sys.exit(main())
