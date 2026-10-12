#!/usr/bin/env python3
"""
gta_noise.py - converte TODAS as texturas do GTA SA (PC/D3D9) em ruido preto e branco,
para o efeito "invisivel parado, visivel em movimento".

Le:   todos os .img (gta3.img, gta_int.img, player.img...) e .txd soltos do jogo
Grava: <jogo>/modloader/noise_mode/...  (nao mexe nos arquivos originais)

Uso:
    pip install numpy
    python gta_noise.py "C:/Games/GTA San Andreas"
    python gta_noise.py "C:/Games/GTA San Andreas" --keep-ui

O que faz com cada textura:
  - troca a cor por ruido 50% preto / 50% branco (pixel a pixel)
  - preserva o alpha (arvores, cercas, particulas) - alpha suave vira pontilhado
  - MANTEM formato, tamanho e mipmaps do original (o jogo le alguns pixels na CPU;
    mudar o formato crasha no loading). Cada mip recebe ruido proprio, sem virar cinza
  - filtro "nearest" para os pontos ficarem nitidos
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np

SECTOR = 2048
RW_STRUCT, RW_EXT, RW_TEXNATIVE, RW_TXD = 0x01, 0x03, 0x15, 0x16
FOURCC = lambda s: struct.unpack("<I", s)[0]
DXT1, DXT3, DXT5 = FOURCC(b"DXT1"), FOURCC(b"DXT3"), FOURCC(b"DXT5")

UI_TXDS = {"fonts.txd", "hud.txd", "fronten1.txd", "fronten2.txd", "fronten3.txd",
           "fronten_pc.txd", "splash1.txd", "splash2.txd", "splash3.txd"}


# ---------------------------------------------------------------- RenderWare
def chunk(buf, pos):
    t, s, v = struct.unpack_from("<III", buf, pos)
    return t, s, v, pos + 12


def mk(t, body, ver):
    return struct.pack("<III", t, len(body), ver) + body


# ---------------------------------------------------------------- alpha decode
def _blocks(data, w, h, bsize):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    b = np.frombuffer(data, np.uint8, bw * bh * bsize).reshape(bh, bw, bsize)
    return b, bw, bh


def _unblock(px, bw, bh, w, h):
    img = px.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    return img[:h, :w]


def decode_alpha(d3dfmt, w, h, data, pal):
    if d3dfmt == DXT1:
        b, bw, bh = _blocks(data, w, h, 8)
        c0 = b[..., 0].astype(np.uint16) | (b[..., 1].astype(np.uint16) << 8)
        c1 = b[..., 2].astype(np.uint16) | (b[..., 3].astype(np.uint16) << 8)
        idx = b[..., 4:8].copy().view("<u4")[..., 0]
        ind = (idx[..., None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3
        a = np.where((c0 <= c1)[..., None] & (ind == 3), 0, 255).astype(np.uint8)
        return _unblock(a, bw, bh, w, h)

    if d3dfmt == DXT3:
        b, bw, bh = _blocks(data, w, h, 16)
        v = b[..., 0:8].copy().view("<u8")[..., 0]
        a = ((v[..., None] >> (np.arange(16, dtype=np.uint64) * 4)) & 0xF) * 17
        return _unblock(a.astype(np.uint8), bw, bh, w, h)

    if d3dfmt == DXT5:
        b, bw, bh = _blocks(data, w, h, 16)
        a0, a1 = b[..., 0].astype(np.int32), b[..., 1].astype(np.int32)
        bits = np.zeros((bh, bw, 8), np.uint8)
        bits[..., :6] = b[..., 2:8]
        v = bits.view("<u8")[..., 0]
        ind = ((v[..., None] >> (np.arange(16, dtype=np.uint64) * 3)) & 7).astype(np.int64)
        gt = a0 > a1
        p = np.zeros((bh, bw, 8), np.int32)
        p[..., 0], p[..., 1] = a0, a1
        for i in range(2, 8):
            p[..., i] = ((8 - i) * a0 + (i - 1) * a1) // 7
        for i in range(2, 6):
            p[..., i] = np.where(gt, p[..., i], ((6 - i) * a0 + (i - 1) * a1) // 5)
        p[..., 6] = np.where(gt, p[..., 6], 0)
        p[..., 7] = np.where(gt, p[..., 7], 255)
        a = np.take_along_axis(p, ind, axis=2).astype(np.uint8)
        return _unblock(a, bw, bh, w, h)

    n = w * h
    if pal is not None:  # P8 / P4
        if len(data) >= n:
            ix = np.frombuffer(data, np.uint8, n)
        else:
            raw = np.frombuffer(data, np.uint8)
            ix = np.stack([raw & 0xF, raw >> 4], 1).reshape(-1)[:n]
        return pal[ix, 3].reshape(h, w)
    if d3dfmt == 21:   # A8R8G8B8
        return np.frombuffer(data, np.uint8, n * 4).reshape(h, w, 4)[..., 3].copy()
    if d3dfmt == 25:   # A1R5G5B5
        v = np.frombuffer(data, "<u2", n).reshape(h, w)
        return np.where(v & 0x8000, 255, 0).astype(np.uint8)
    if d3dfmt == 26:   # A4R4G4B4
        v = np.frombuffer(data, "<u2", n).reshape(h, w)
        return ((v >> 12) * 17).astype(np.uint8)
    if d3dfmt == 28:   # A8
        return np.frombuffer(data, np.uint8, n).reshape(h, w).copy()
    if d3dfmt == 51:   # A8L8
        return (np.frombuffer(data, "<u2", n).reshape(h, w) >> 8).astype(np.uint8)
    return np.full((h, w), 255, np.uint8)  # formatos sem alpha


# ---------------------------------------------------------------- encode (mesmo formato do original)
def _to_blocks(ind, w, h):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    pad = np.zeros((bh * 4, bw * 4), ind.dtype)
    pad[:h, :w] = ind
    return pad.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16)


def _pack(blk, bits, dtype):
    sh = np.arange(16, dtype=dtype) * bits
    return (blk.astype(dtype) << sh).sum(axis=2, dtype=dtype)


def encode(d3dfmt, w, h, white, opaque, paletted, size):
    """Retorna os pixels no MESMO formato/tamanho do original, ou None se desconhecido."""
    n = w * h
    vis = white & opaque
    if paletted:
        ix = np.where(~opaque, 2, np.where(white, 1, 0)).astype(np.uint8).reshape(-1)
        if size >= n:
            out = ix.tobytes()
        else:  # 4 bits por pixel
            ix = np.concatenate([ix, np.zeros(len(ix) % 2, np.uint8)])
            out = (ix[0::2] | (ix[1::2] << 4)).astype(np.uint8).tobytes()
    elif d3dfmt == DXT1:
        ind = np.where(~opaque, 3, np.where(white, 1, 0))  # c0=preto <= c1=branco: modo 3 cores
        blk = _to_blocks(ind, w, h)
        o = np.empty(blk.shape[:2] + (2,), "<u4")
        o[..., 0] = 0xFFFF0000
        o[..., 1] = _pack(blk, 2, np.uint32)
        out = o.tobytes()
    elif d3dfmt in (DXT3, DXT5):
        col = _to_blocks(np.where(white, 0, 1), w, h)  # c0=branco > c1=preto: idx0 branco, idx1 preto
        o = np.empty(col.shape[:2] + (4,), "<u4")
        if d3dfmt == DXT3:
            al = _pack(_to_blocks(np.where(opaque, 0xF, 0), w, h), 4, np.uint64)
            o[..., 0] = (al & 0xFFFFFFFF).astype(np.uint32)
            o[..., 1] = (al >> np.uint64(32)).astype(np.uint32)
        else:  # DXT5: a0=255 a1=0, idx0 -> 255, idx1 -> 0
            ai = _pack(_to_blocks(np.where(opaque, 0, 1), w, h), 3, np.uint64)
            lo = (np.uint64(0x00FF) | ((ai & np.uint64(0xFFFF)) << np.uint64(16)))
            o[..., 0] = lo.astype(np.uint32)
            o[..., 1] = (ai >> np.uint64(16)).astype(np.uint32)
        o[..., 2] = 0x0000FFFF
        o[..., 3] = _pack(col, 2, np.uint32)
        out = o.tobytes()
    else:
        v8 = white.astype(np.uint8) * 255
        a8 = opaque.astype(np.uint8) * 255
        if d3dfmt == 21:   # A8R8G8B8
            out = np.stack([v8, v8, v8, a8], -1).tobytes()
        elif d3dfmt == 22:  # X8R8G8B8
            out = np.stack([v8, v8, v8, np.full_like(v8, 255)], -1).tobytes()
        elif d3dfmt == 20:  # R8G8B8
            out = np.stack([v8, v8, v8], -1).tobytes()
        elif d3dfmt == 23:  # R5G6B5
            out = np.where(white, 0xFFFF, 0).astype("<u2").tobytes()
        elif d3dfmt == 24:  # X1R5G5B5
            out = np.where(white, 0xFFFF, 0x8000).astype("<u2").tobytes()
        elif d3dfmt == 25:  # A1R5G5B5
            out = (np.where(white, 0x7FFF, 0) | np.where(opaque, 0x8000, 0)).astype("<u2").tobytes()
        elif d3dfmt == 26:  # A4R4G4B4
            out = (np.where(white, 0x0FFF, 0) | np.where(opaque, 0xF000, 0)).astype("<u2").tobytes()
        elif d3dfmt == 28:  # A8
            out = (vis.astype(np.uint8) * 255).tobytes()
        elif d3dfmt == 50:  # L8
            out = v8.tobytes()
        elif d3dfmt == 51:  # A8L8
            out = (v8.astype("<u2") | (a8.astype("<u2") << 8)).astype("<u2").tobytes()
        else:
            return None
    # seguranca: nunca muda o tamanho do bloco de pixels
    if len(out) < size:
        out += bytes(size - len(out))
    return out[:size]


# ---------------------------------------------------------------- conversao
def convert_native(body, ver, seed):
    t, s, sv, p = chunk(body, 0)
    st, tail = body[p:p + s], body[p + s:]
    platform = struct.unpack_from("<I", st, 0)[0]
    if platform != 9:
        return body, False

    (filt, name, mask, rfmt, d3dfmt, w, h,
     depth, levels, rtype, flags) = struct.unpack_from("<I32s32sIIHHBBBB", st, 4)
    hdr = bytearray(st[:88])
    off = 88
    pal, pal_n = None, 0
    if rfmt & 0x2000:
        pal_n = 256
    elif rfmt & 0x4000:
        pal_n = 32
    if pal_n:
        pal = np.frombuffer(st, np.uint8, pal_n * 4, off).reshape(pal_n, 4)
        off += pal_n * 4

    lv = []
    for _ in range(max(1, levels)):
        if off + 4 > len(st):
            break
        size = struct.unpack_from("<I", st, off)[0]
        lv.append((off + 4, size))
        off += 4 + size
    extra = st[off:]
    if not lv:
        return body, False

    try:
        alpha0 = decode_alpha(d3dfmt, w, h, st[lv[0][0]:lv[0][0] + lv[0][1]], pal) \
            if flags & 1 else np.full((h, w), 255, np.uint8)
    except Exception:
        alpha0 = np.full((h, w), 255, np.uint8)

    rng = np.random.default_rng(zlib.crc32(name.split(b"\0")[0].lower()) ^ seed)
    new_levels = []
    for i, (doff, size) in enumerate(lv):
        lw, lh = max(1, w >> i), max(1, h >> i)
        a = alpha0[::1 << i, ::1 << i][:lh, :lw]
        if a.shape != (lh, lw):
            a = np.pad(a, ((0, lh - a.shape[0]), (0, lw - a.shape[1])), mode="edge")
        white = rng.random((lh, lw)) < 0.5
        opaque = rng.random((lh, lw)) * 255 < a
        px = encode(d3dfmt, lw, lh, white, opaque, pal is not None, size)
        if px is None:
            return body, False  # formato desconhecido: deixa a textura original
        new_levels.append(struct.pack("<I", size) + px)

    # filtro nearest (1) ou mip-nearest (3); todo o resto do cabecalho fica igual
    hdr[4] = 3 if len(lv) > 1 else 1
    new_pal = b""
    if pal is not None:
        npal = np.zeros((pal_n, 4), np.uint8)
        npal[0] = (0, 0, 0, 255)
        npal[1] = (255, 255, 255, 255)
        npal[2] = (0, 0, 0, 0)
        new_pal = npal.tobytes()
    st_n = bytes(hdr) + new_pal + b"".join(new_levels) + extra
    return mk(RW_STRUCT, st_n, sv) + tail, True


def process_txd(buf, seed):
    t, s, ver, p = chunk(buf, 0)
    if t != RW_TXD:
        raise ValueError("nao e um TXD")
    end = p + s
    t, s, v, q = chunk(buf, p)
    _, dev = struct.unpack_from("<HH", buf, q)
    p = q + s
    natives, rest, n = [], b"", 0
    while p < end:
        t, s, v, q = chunk(buf, p)
        if t == RW_TEXNATIVE:
            out, ok = convert_native(buf[q:q + s], v, seed)
            natives.append(mk(RW_TEXNATIVE, out, v)); n += ok
        else:
            rest += buf[p:q + s]
        p = q + s
    st = mk(RW_STRUCT, struct.pack("<HH", len(natives), dev), ver)
    return mk(RW_TXD, st + b"".join(natives) + rest, ver), n


# ---------------------------------------------------------------- arquivos
def iter_img(path):
    with open(path, "rb") as f:
        if f.read(4) != b"VER2":
            print(f"  [pulando] {path} nao e IMG v2"); return
        count = struct.unpack("<I", f.read(4))[0]
        entries = []
        for _ in range(count):
            off, ss, asz, name = struct.unpack("<IHH24s", f.read(32))
            entries.append((off, (ss or asz) * SECTOR, name.split(b"\0")[0].decode("latin1")))
        for off, size, name in entries:
            if name.lower().endswith(".txd"):
                f.seek(off * SECTOR)
                yield name, f.read(size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", help="pasta do GTA San Andreas")
    ap.add_argument("--out", default=None, help="padrao: <jogo>/modloader/noise_mode")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep-ui", action="store_true", help="nao converte HUD/menu/fontes")
    a = ap.parse_args()

    out_root = a.out or os.path.join(a.game, "modloader", "noise_mode")
    skip = os.path.normcase(os.path.join(a.game, "modloader"))
    stats = {"txd": 0, "tex": 0, "err": 0}

    def handle(name, data, dest_dir):
        if a.keep_ui and name.lower() in UI_TXDS:
            return
        try:
            new, n = process_txd(data, a.seed)
        except Exception as e:
            stats["err"] += 1
            print(f"  [erro] {name}: {e}")
            return
        os.makedirs(dest_dir, exist_ok=True)
        with open(os.path.join(dest_dir, name), "wb") as f:
            f.write(new)
        stats["txd"] += 1; stats["tex"] += n

    for root, dirs, files in os.walk(a.game):
        if os.path.normcase(root).startswith(skip):
            dirs[:] = []; continue
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
