"""
Almacenamiento de fotos: Cloudflare R2 (produccion) o una carpeta local (pruebas).
Misma interfaz para ambos, asi el resto del codigo no cambia.
"""
from __future__ import annotations

import os
import shutil
from typing import BinaryIO, Iterable, List, Optional, Tuple, Union

Datos = Union[bytes, BinaryIO]


class Almacen:
    def listar(self, prefijo: str = "") -> List[Tuple[str, int]]: ...
    def subir(self, key: str, datos: Datos, tipo: str = "image/jpeg") -> None: ...
    def leer(self, key: str) -> bytes: ...
    def descargar_a(self, key: str, destino: BinaryIO) -> None: ...
    def copiar(self, origen: str, destino: str) -> None: ...
    def borrar(self, keys: Iterable[str]) -> None: ...
    def existe(self, key: str) -> bool: ...

    def borrar_prefijo(self, prefijo: str) -> int:
        keys = [k for k, _ in self.listar(prefijo)]
        self.borrar(keys)
        return len(keys)


class AlmacenR2(Almacen):
    def __init__(self, endpoint: str, key_id: str, secret: str, bucket: str):
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=key_id, aws_secret_access_key=secret,
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                          max_pool_connections=32),
        )

    def listar(self, prefijo: str = "") -> List[Tuple[str, int]]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefijo):
            out += [(o["Key"], o.get("Size", 0)) for o in page.get("Contents", []) if not o["Key"].endswith("/")]
        return out

    def subir(self, key: str, datos: Datos, tipo: str = "image/jpeg") -> None:
        if isinstance(datos, (bytes, bytearray)):
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=bytes(datos), ContentType=tipo)
        else:
            self.s3.upload_fileobj(datos, self.bucket, key, ExtraArgs={"ContentType": tipo})

    def leer(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def descargar_a(self, key: str, destino: BinaryIO) -> None:
        self.s3.download_fileobj(self.bucket, key, destino)

    def copiar(self, origen: str, destino: str) -> None:
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": origen}, Key=destino)

    def borrar(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        for i in range(0, len(keys), 1000):
            lote = [{"Key": k} for k in keys[i:i + 1000]]
            if lote:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": lote, "Quiet": True})

    def existe(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class AlmacenLocal(Almacen):
    """Carpeta en disco con la misma estructura de 'keys' que R2 (para probar sin nube)."""

    def __init__(self, raiz: str):
        self.raiz = os.path.abspath(raiz)
        os.makedirs(self.raiz, exist_ok=True)

    def _ruta(self, key: str) -> str:
        partes = [p for p in key.replace("\\", "/").split("/") if p not in ("", ".", "..")]
        ruta = os.path.abspath(os.path.join(self.raiz, *partes))
        if not ruta.startswith(self.raiz):
            raise ValueError("ruta invalida")
        return ruta

    def listar(self, prefijo: str = "") -> List[Tuple[str, int]]:
        out = []
        for dirpath, _, files in os.walk(self.raiz):
            for f in files:
                p = os.path.join(dirpath, f)
                k = os.path.relpath(p, self.raiz).replace("\\", "/")
                if k.startswith(prefijo):
                    out.append((k, os.path.getsize(p)))
        return sorted(out)

    def subir(self, key: str, datos: Datos, tipo: str = "image/jpeg") -> None:
        p = self._ruta(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            if isinstance(datos, (bytes, bytearray)):
                f.write(datos)
            else:
                shutil.copyfileobj(datos, f)

    def leer(self, key: str) -> bytes:
        with open(self._ruta(key), "rb") as f:
            return f.read()

    def descargar_a(self, key: str, destino: BinaryIO) -> None:
        with open(self._ruta(key), "rb") as f:
            shutil.copyfileobj(f, destino)

    def copiar(self, origen: str, destino: str) -> None:
        p = self._ruta(destino)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(self._ruta(origen), p)

    def borrar(self, keys: Iterable[str]) -> None:
        for k in keys:
            try:
                os.remove(self._ruta(k))
            except FileNotFoundError:
                pass
        for dirpath, dirs, files in sorted(os.walk(self.raiz), key=lambda x: -len(x[0])):
            if dirpath != self.raiz and not dirs and not files:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass

    def existe(self, key: str) -> bool:
        return os.path.exists(self._ruta(key))


def crear_almacen() -> Optional[Almacen]:
    ak, sk, ep = os.getenv("R2_ACCESS_KEY_ID", ""), os.getenv("R2_SECRET_ACCESS_KEY", ""), os.getenv("R2_ENDPOINT_URL", "")
    if ak and sk and ep:
        return AlmacenR2(ep, ak, sk, os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos"))
    local = os.getenv("ALMACEN_LOCAL", "")
    if local:
        return AlmacenLocal(local)
    return None
