# Agente 1 Clasificador — estado y pendientes (21 nov 2026, 08:10)

## >>> DONDE RETOMAR EN LA PROXIMA SESION <<<
Pídele a Claude: "Lee PENDIENTES.md del agente1-clasificador y continúa desde 'Siguiente paso'."

### Hecho y subido a GitHub (main)
- 05f12be Florence-2 + votación · f8cfa01 índice de caras en el ZIP (00_INDICE_JUGADORES.jpg + 00_ROSTRO.jpg por carpeta).
- 59b4880 PRIVACIDAD: las huellas faciales se borran al terminar cada partido (nube y memoria). /api/diagnostico
  muestra "huellas_guardadas" (debe ser 0 tras terminar). Cuadrícula "Ponle número a cada cara" solo como respaldo.
- 630ff4e arreglo del Dockerfile: 05f12be y f8cfa01 FALLARON al construir en Render (pip rechazaba metadatos de
  typing-extensions del índice de pytorch). Render siguió con 5489212 (funciona). Revisar que 630ff4e quede "live".

### Decisión del usuario (no negociable)
- NADA de banco de caras entre partidos (riesgo legal/extorsión en Venezuela, menores). Cero datos biométricos guardados.
- El agente debe nombrar las carpetas solo; el trabajo manual no es solución.

### Diagnóstico clave (medido hoy)
- En las carpetas "sin número" el número SÍ se ve grande y claro en muchas fotos (8 La Guaira, 3/6/10 Trujillo
  rojos, 4 Caracas, 6 Apure). El fallo es del LECTOR, no de las fotos. Hojas: scratchpad/out/sin_num_cliente_*.jpg.
- Florence-2 base y PP-OCR (RapidOCR) NO leen esos números ni con recorte justo (ver scratchpad/exp_tight.py,
  exp_rapid2.py). Son lectores de texto: ignoran un número grande y decorado.
- En prueba: SmolVLM-256M-Instruct (Apache 2.0) preguntando "¿qué número tiene la camiseta?" (exp_smol.py).

### Siguiente paso
1. Lector de números que SÍ lea dígitos grandes decorados. Opciones a medir sobre out/tight/*.jpg:
   a) SmolVLM-256M/500M (preguntar el número; desactivar división de imagen para velocidad).
   b) Clasificador de dígitos entrenado (estilo SVHN) sobre el recorte del número.
   Aceptar solo si acierta los casos de out/tight sin inventar números en recortes sin número.
2. Integrarlo como 3er votante en _votar_florence (motor_biometrico.py) y medir 20 + 40 fotos.
3. Probar en Render en vivo con las 20 fotos del cliente.

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
