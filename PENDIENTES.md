# Agente 1 Clasificador — estado y pendientes (21 nov 2026)

## >>> DONDE RETOMAR EN LA PROXIMA SESION <<<
Pídele a Claude: "Lee PENDIENTES.md del agente1-clasificador y continúa desde el paso 2."

Estado al cerrar (21 nov 2026, ~01:00):
- EN VIVO y funcionando: commit 86b4336 (sin Florence-2). No se ha publicado nada después.
- EN LOCAL, SIN PUBLICAR (sin commit):
  - `lector_florence.py` (nuevo): lector Florence-2 local.
  - `motor_biometrico.py`: función `_votar_florence` (jurado CRNN + Florence-2 con reglas de votación).
  - `identidad.py`: función `equipos_en_texto` (nombres de equipo dentro del texto OCR).
  - Falta: probarlo con las 20 + 40 fotos, añadir torch/transformers al Dockerfile y requirements, publicar.
- Resultado del jurado en casos difíciles (Florence-2 base con 4 variantes de imagen): no rescató
  ninguno de los 6 primeros (4 Caracas, 6 Apure, 8 La Guaira espaldas, 14 Sucre, 3 y 6 rojos de Trujillo).
  Conclusión: para números rojos grandes de espaldas hace falta la "huella visual del dorsal" (paso 4),
  no más lectores de texto.
- Verificado: torch 2.14.1+cpu y transformers 5.20.2 instalan en Python 3.10 Linux (Render).

## En vivo (commit 86b4336, Render 2 CPU / 4 GB, 85 USD/mes)
- Motor biométrico: YOLOX (personas) + YuNet/SFace (huella facial) + PP-OCR DB + CRNN (dorsales) + color/texto (equipo).
- Fusión de identidades con candados; medidor de certeza; dudas "mismo", "quien", "unir" con un toque.
- Fichas para protagonistas de una sola foto (partidos de 40-50 fotos).
- Estado compartido en R2, reanudar tras caída ("Continuar"), análisis guardado por foto.
- Lector con Claude (lector_vision.py) listo pero APAGADO: el usuario no puede pagar la API (solo su plan de 20 USD).

## Mediciones con las 20 fotos del cliente (25 números visibles)
| Lector | Leídos bien | Errores |
|---|---|---|
| CRNN actual | 7/25 (28 %) | 4 (6, 5, 71, 6) |
| Florence-2 base (MIT, local, gratis) | 13/25 (52 %) | 2 (2, 21) |
| Ambos con votación | 14/25 (56 %) | 0 aceptados |
- Florence-2 base: 9,7 s por torso en CPU, +955 MB RAM, pico total 2,2 GB.
- Lee nombres de equipo: Caracas, La Guaira, Trujillo, Carabobo.
- Casos que ningún lector gratuito lee: números rojos grandes de espaldas de Trujillo (3, 6, 10, 7), 4 Caracas, 14 Sucre, 17 Carabobo, 6 Apure, 8 La Guaira de espaldas.

## Pendientes en orden
1. Terminar `exp_jurado.py` (Florence base/large × variantes: original, contraste, ampliada, número aislado por color) sobre los 11 casos difíciles.
2. Integrar Florence-2 + votación en `motor_biometrico.py`:
   - Usarlo solo en protagonistas sin lectura CRNN segura (velocidad).
   - Regla: 2+ lectores coinciden = seguro; 1 solo = duda; 2 números en un recorte o 2 equipos en el texto = no asignar.
   - Mismo número en 2 jugadores de una foto: gana el de más votos.
   - Dockerfile: añadir torch CPU + transformers + descarga de `florence-community/Florence-2-base` (nativo, sin trust_remote_code).
3. Batería de pruebas en vivo: dudas/corregir/unir/ZIP, reinicio y "Continuar", dos partidos seguidos, celular.
4. Casos difíciles sin leer el número:
   - Huella visual del dorsal (mismo glifo/colores entre fotos ⇒ mismo jugador).
   - Huella corporal de espaldas (peinado, zapatos, medias, muñequeras) para sugerencias "¿Quién es?".
5. Opcional: bajar Render a 25 USD/mes si no se integra Florence-2 (2 GB bastan sin él).

## Pruebas locales (scratchpad de la sesión; se puede rehacer)
- `fotos-cliente-real\` (20 fotos del cliente bajadas de la bóveda).
- Entorno: Python 3.14 + opencv-python-headless 4.11.0.86 (igual que Render), torch 2.15 CPU, transformers 5.23.
