"""
services/hf_storage.py — Tiện ích push/load private folder data lên/về từ Hugging Face.

Hỗ trợ cả import trực tiếp dưới dạng hàm python và chạy như CLI độc lập.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, snapshot_download

# Tự động load file .env từ thư mục gốc của project
# Thử tìm file .env tại thư mục hiện tại hoặc các thư mục cha
def init_env():
    """Tải các biến môi trường từ file .env nếu có."""
    current_path = Path(__file__).resolve()
    # Tìm lên tối đa 4 cấp thư mục cha để tìm file .env
    for _ in range(4):
        env_path = current_path.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
        current_path = current_path.parent
    else:
        load_dotenv()


def get_token(token: str | None = None) -> str:
    """Trả về token Hugging Face từ tham số đầu vào hoặc biến môi trường."""
    if token:
        return token
    
    # Đảm bảo env đã được load
    init_env()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Không tìm thấy token Hugging Face. Vui lòng cấu hình biến môi trường 'HF_TOKEN' "
            "trong file .env hoặc truyền trực tiếp qua tham số/CLI."
        )
    return hf_token


def push_folder_to_huggingface(
    folder_path: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Push toàn bộ cấu trúc thư mục local lên Hugging Face Dataset repository (private mặc định).

    Args:
        folder_path: Đường dẫn tới thư mục local cần upload.
        repo_id: ID repository trên Hugging Face (ví dụ: 'viamr-project/diverse-reasoning-data').
        token: Token Hugging Face (nếu không truyền, tự động đọc từ .env).
        private: Tạo repo ở chế độ private hay public (mặc định True).
        ignore_patterns: Danh sách file pattern cần bỏ qua không upload (ví dụ: ['*.tmp', '.DS_Store']).

    Returns:
        Đường dẫn URL của repository trên Hugging Face.
    """
    init_env()
    token = get_token(token)
    
    local_path = Path(folder_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Thư mục local '{folder_path}' không tồn tại.")
    if not local_path.is_dir():
        raise ValueError(f"Đường dẫn '{folder_path}' không phải là một thư mục.")

    print(f"[*] Đang khởi tạo/kiểm tra repository '{repo_id}' trên Hugging Face...")
    # Tạo repository nếu chưa tồn tại
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    print(f"[*] Đang upload thư mục '{local_path.resolve()}' lên '{repo_id}'...")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=ignore_patterns,
    )
    
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[✓] Đã upload thành công thư mục lên: {url}")
    return url


def load_folder_from_huggingface(
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """
    Tải toàn bộ thư mục từ Hugging Face Dataset repository về máy local, giữ nguyên cấu trúc định dạng.

    Args:
        repo_id: ID repository trên Hugging Face (ví dụ: 'viamr-project/diverse-reasoning-data').
        local_dir: Đường dẫn thư mục local muốn lưu kết quả về.
        token: Token Hugging Face (nếu không truyền, tự động đọc từ .env).
        ignore_patterns: Danh sách file pattern cần bỏ qua không tải về.

    Returns:
        Đường dẫn Path tới thư mục local đã tải về thành công.
    """
    init_env()
    token = get_token(token)
    
    dest_path = Path(local_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang tải dữ liệu từ repository '{repo_id}' về thư mục '{dest_path.resolve()}'...")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_path),
        token=token,
        ignore_patterns=ignore_patterns,
    )
    
    print(f"[✓] Đã tải dữ liệu thành công về thư mục: {dest_path.resolve()}")
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đồng bộ hóa thư mục dữ liệu với Hugging Face Dataset Repository."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Các lệnh được hỗ trợ")

    # Subparser cho lệnh push
    push_parser = subparsers.add_parser("push", help="Push thư mục local lên Hugging Face")
    push_parser.add_argument(
        "--folder",
        default="data",
        help="Đường dẫn thư mục local cần upload (mặc định: 'data')",
    )
    push_parser.add_argument(
        "--repo-id",
        default="viamr-project/diverse-reasoning-data",
        help="ID repository trên HF (mặc định: 'viamr-project/diverse-reasoning-data')",
    )
    push_parser.add_argument(
        "--public",
        action="store_true",
        help="Đặt repository ở chế độ Public thay vì Private mặc định",
    )
    push_parser.add_argument(
        "--ignore",
        nargs="+",
        help="Các file pattern cần bỏ qua không upload (ví dụ: *.raw_samples.jsonl)",
    )
    push_parser.add_argument(
        "--token",
        help="Token Hugging Face ghi đè giá trị trong file .env",
    )

    # Subparser cho lệnh load
    load_parser = subparsers.add_parser("load", help="Tải thư mục từ Hugging Face về local")
    load_parser.add_argument(
        "--repo-id",
        default="viamr-project/diverse-reasoning-data",
        help="ID repository trên HF (mặc định: 'viamr-project/diverse-reasoning-data')",
    )
    load_parser.add_argument(
        "--folder",
        default="data",
        help="Đường dẫn thư mục local để tải về (mặc định: 'data')",
    )
    load_parser.add_argument(
        "--ignore",
        nargs="+",
        help="Các file pattern cần bỏ qua không tải về",
    )
    load_parser.add_argument(
        "--token",
        help="Token Hugging Face ghi đè giá trị trong file .env",
    )

    args = parser.parse_args()

    try:
        if args.command == "push":
            push_folder_to_huggingface(
                folder_path=args.folder,
                repo_id=args.repo_id,
                token=args.token,
                private=not args.public,
                ignore_patterns=args.ignore,
            )
        elif args.command == "load":
            load_folder_from_huggingface(
                repo_id=args.repo_id,
                local_dir=args.folder,
                token=args.token,
                ignore_patterns=args.ignore,
            )
    except Exception as e:
        print(f"[x] Lỗi: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()