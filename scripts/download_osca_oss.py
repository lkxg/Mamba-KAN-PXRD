#!/usr/bin/env python3
"""Download files from OSCA-OSS through its S3-compatible API.

Credentials are read from environment variables by default:
  OSCA_ACCESS_KEY_ID / OSCA_SECRET_ACCESS_KEY
or:
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

try:
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    BotoCoreError = ClientError = RuntimeError
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_ENDPOINT = "https://fgws3-ocloud.ihep.ac.cn"
DEFAULT_REGION = "us-east-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and download objects from an OSCA-OSS bucket."
    )
    parser.add_argument("--bucket", required=True, help="OSCA-OSS bucket name.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OSCA_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"S3 endpoint URL. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("OSCA_REGION", DEFAULT_REGION),
        help=f"S3 region name. Default: {DEFAULT_REGION}",
    )
    parser.add_argument(
        "--access-key-id",
        default=os.getenv("OSCA_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"),
        help="AccessKeyId. Prefer setting OSCA_ACCESS_KEY_ID instead.",
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.getenv("OSCA_SECRET_ACCESS_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        help="AccessKeySecret. Prefer setting OSCA_SECRET_ACCESS_KEY instead.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Remote prefix to list/download, for example 'dataset/raw/'.",
    )
    parser.add_argument(
        "--key",
        help="Download one exact object key. If omitted, downloads all objects under --prefix.",
    )
    parser.add_argument(
        "--output",
        default="downloads/osca_oss",
        help=(
            "Output file path for --key, or output directory for prefix downloads. "
            "Default: downloads/osca_oss"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list matched remote objects; do not download.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Keep the full remote prefix under the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files that already exist.",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL. Usually only needed when endpoint starts with http://.",
    )
    args = parser.parse_args()

    if not args.access_key_id or not args.secret_access_key:
        parser.error(
            "missing credentials: set OSCA_ACCESS_KEY_ID and "
            "OSCA_SECRET_ACCESS_KEY, or pass --access-key-id and "
            "--secret-access-key"
        )
    if args.key and args.list:
        parser.error("--key and --list cannot be used together")
    return args


def make_client(args: argparse.Namespace):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is not installed. Run: pip install -r requirements.txt"
        ) from exc

    use_ssl = args.endpoint.startswith("https://") and not args.no_ssl
    return boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key_id,
        aws_secret_access_key=args.secret_access_key,
        region_name=args.region,
        use_ssl=use_ssl,
    )


def iter_objects(s3_client, bucket: str, prefix: str) -> Iterator[dict]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                yield obj


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        size /= 1024
    return f"{num_bytes} B"


def local_path_for_key(
    output: Path, key: str, prefix: str, keep_prefix: bool, single_key: bool
) -> Path:
    if single_key:
        if output.suffix:
            return output
        return output / Path(key).name

    relative_key = key if keep_prefix else key.removeprefix(prefix).lstrip("/")
    if not relative_key:
        relative_key = Path(key).name
    return output / relative_key


def download_object(
    s3_client,
    bucket: str,
    key: str,
    destination: Path,
    size: int,
    overwrite: bool,
) -> bool:
    if destination.exists() and not overwrite:
        print(f"skip existing: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    if tqdm is None:
        s3_client.download_file(bucket, key, str(destination))
    else:
        progress = tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=Path(key).name or key,
            leave=False,
        )
        try:
            s3_client.download_file(
                bucket,
                key,
                str(destination),
                Callback=lambda chunk: progress.update(chunk),
            )
        finally:
            progress.close()

    print(f"downloaded: s3://{bucket}/{key} -> {destination}")
    return True


def main() -> int:
    args = parse_args()
    s3_client = make_client(args)

    try:
        if args.list:
            objects = list(iter_objects(s3_client, args.bucket, args.prefix))
            if not objects:
                print(f"no objects found: s3://{args.bucket}/{args.prefix}")
                return 0
            for obj in objects:
                print(f"{obj['Key']}\t{format_size(obj['Size'])}")
            print(f"total: {len(objects)} objects")
            return 0

        output = Path(args.output)
        if args.key:
            head = s3_client.head_object(Bucket=args.bucket, Key=args.key)
            destination = local_path_for_key(
                output, args.key, args.prefix, args.keep_prefix, single_key=True
            )
            downloaded = download_object(
                s3_client,
                args.bucket,
                args.key,
                destination,
                head["ContentLength"],
                args.overwrite,
            )
            return 0 if downloaded or destination.exists() else 1

        objects = list(iter_objects(s3_client, args.bucket, args.prefix))
        if not objects:
            print(f"no objects found: s3://{args.bucket}/{args.prefix}")
            return 0

        downloaded_count = 0
        for obj in objects:
            destination = local_path_for_key(
                output,
                obj["Key"],
                args.prefix,
                args.keep_prefix,
                single_key=False,
            )
            if download_object(
                s3_client,
                args.bucket,
                obj["Key"],
                destination,
                obj["Size"],
                args.overwrite,
            ):
                downloaded_count += 1

        print(f"done: downloaded {downloaded_count}/{len(objects)} objects")
        return 0
    except (RuntimeError, BotoCoreError, ClientError) as exc:
        print(f"OSCA-OSS request failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
