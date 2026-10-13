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

Pra desinstalar basta apagar a pasta noise_mode de dentro de modloader.

EDIT:
Adicionado o GTA_SOLID.py que deixa o jogo com uma cor lisa em tom bege.
